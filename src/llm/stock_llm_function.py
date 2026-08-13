
import sys
import re
import json
import datetime
from decimal import Decimal

from sqlalchemy import text
import ollama

from src.common.setup_log import SetupLogger
from src.database.connect_postgres import PostgresDB

OLLAMA_MODEL = "qwen2.5:1.5b"  # 경량/고속, SQL 생성 성능 우수 (대안: gemma3:1b,qwen2.5:0.5b )
DEFAULT_LIMIT = 20  # 쿼리에 LIMIT이 없을 때 강제로 붙이는 최대 행 수
MAX_LIMIT = 100     # 허용하는 최대 행 수 (LLM이 큰 값을 써도 이 값으로 제한)

# 실행을 차단할 위험 키워드 (SELECT 이외의 조작/DDL/DML)
FORBIDDEN_KEYWORDS = (
    "insert", "update", "delete", "drop", "alter", "create",
    "truncate", "grant", "revoke", "merge", "call", "copy",
)

# LLM이 SQL 뒤에 덧붙이는 해설 줄을 판별하는 패턴
# (예: "### 설명", "-- 이 쿼리는...", "**참고**", "Note: ...")
EXPLANATION_LINE_PATTERN = re.compile(
    r"^\s*(#{1,6}(\s|$)|--|/\*|\*\*|설명|참고|주의|Explanation|Note)",
    flags=re.IGNORECASE,
)


class StockLLMAnalysis:
    """
    자연어 질문을 SQL로 변환해 주가 데이터를 조회하는 클래스 (Text-to-SQL)

    LLM은 오직 SELECT 쿼리 생성만 담당하고, 실행 전 코드가 안전성을 검증한다.
    """

    # LLM에게 알려줄 테이블 스키마 (Context) — 환각 방지의 핵심
    SCHEMA_GUIDE = """[테이블] t_stock_price_data (일별 주가 + 계산지표)
                      [컬럼]
                            - trade_date   (date) : 거래일. 날짜 비교/정렬/INTERVAL 연산 가능
                                           (예: trade_date >= '2026-07-01',
                                           trade_date >= (SELECT MAX(trade_date) ...) - INTERVAL '7 day')
                            - ticker_name  (문자열) : 종목명 (예: '경농', '삼성전자')
                            - ticker_code  (문자열) : 종목코드
                            - open_price   (숫자) : 시가
                            - high_price   (숫자) : 고가
                            - low_price    (숫자) : 저가
                            - close_price  (숫자) : 종가
                            - volume       (정수) : 거래량
                            - daily_change (숫자, 비율) : 전일대비 등락률 (0.0022 = +0.22%)
                            - ma_20        (숫자) : 20일 이동평균
                            - volatility   (숫자) : 변동성
                            - dd_high      (숫자, 비율) : 고점대비 낙폭 (음수)
                            - ret_low      (숫자, 비율) : 저점대비 수익률 (양수)
                            - del_yn       (불리언) : 삭제 여부. 조회 시 반드시 del_yn = false 조건 포함"""

    def __init__(self, model_name=OLLAMA_MODEL, temperature=0.0):
        """
        Text-to-SQL 클래스 초기 설정

        param model_name: SQL 생성에 사용할 Ollama 모델명.
                          기본값은 모듈 상수 OLLAMA_MODEL
        param temperature: 0 이면 같은 질문에 항상 같은 SQL이 나오고,
                          값이 커지면 매번 달라진다.
                          SQL은 정답이 하나뿐이라 0.0으로 둔다.
                          (값을 넘기지 않으면 Ollama 기본값 0.8이 적용된다)

        """
        self.logger = SetupLogger.get_logger()
        self.db = PostgresDB()

        # 모델과 호출 옵션을 인스턴스 속성으로 보관 (LLMToolCalling과 동일한 방식)
        self.MODEL_NAME = model_name
        self.OPTIONS = {"temperature": temperature}

    def get_latest_trade_date(self):
        """
        DB의 최신 거래일을 조회 (프롬프트에 주입해 LLM의 날짜 지어내기 방지)

        return: str - 최신 거래일 'YYYY-MM-DD'. 조회 실패 시 None
        """
        try:
            with self.db.get_postgres_db() as session:
                row = session.execute(
                    text(
                        "SELECT MAX(trade_date) FROM t_stock_price_data "
                        "WHERE del_yn = false"
                    )
                ).first()
            # date 객체 → 'YYYY-MM-DD' 문자열 (프롬프트 삽입용)
            return str(row[0]) if row and row[0] else None
        except Exception as e:
            self.logger.warning(f"최신 거래일 조회 실패: {e}")
            return None

    def generate_sql(self, question, error_feedback=None):
        """
        자연어 질문을 SELECT 쿼리로 변환 (LLM 호출)

        스키마를 프롬프트에 명시해 LLM이 실제 존재하는 컬럼만 쓰도록 유도하고,
        자주 틀리는 패턴(날짜함수/특정 종목의 최근 N일)은 Few-shot 예시로 교정한다.
        생성 결과는 _extract_sql로 코드펜스와 해설을 걷어내 순수 SQL만 반환한다.

        param question: 사용자 자연어 질문(str)
        param error_feedback: 직전 실행 에러 메시지(str). 재시도 시 LLM에게 전달해 교정 유도

        return: str - LLM이 생성한 SQL 문자열 (검증 전 원본)
        """
        error_block = ""
        if error_feedback:
            error_block = f"""
                            직전에 생성한 SQL이 아래 에러로 실패했습니다. 에러를 참고해 고친 SQL을 만드세요.
                            [에러] {error_feedback}
                            """

        # 실제 최신 거래일을 알려줘 LLM이 날짜를 지어내지 못하게 한다
        latest = self.get_latest_trade_date()
        date_context = (
            f"[중요] 데이터의 최신 거래일은 '{latest}' 입니다. "
            "날짜가 필요하면 이 값이나 MAX(trade_date) 서브쿼리를 쓰고, 임의의 날짜를 지어내지 마세요."
            if latest else ""
        )

        prompt = f"""당신은 PostgreSQL 전문가입니다. 아래 스키마를 참고해 질문에 답하는
                    SELECT 쿼리 하나만 생성하세요. 설명 없이 SQL만 출력하세요.

                {self.SCHEMA_GUIDE}
                
                {date_context}
                
                규칙:
                    - 반드시 SELECT 문만 작성하세요. 데이터를 변경하는 문(INSERT/UPDATE/DELETE/DROP 등)은 금지입니다.
                    - 항상 WHERE 조건에 del_yn = false 를 포함하세요.
                    - INTERVAL 등 날짜 연산은 사용할 수 있습니다. 단 CURRENT_DATE, NOW()는 절대 쓰지 마세요.
                      ('오늘'은 실제 달력 날짜가 아니라 데이터의 최신 거래일 MAX(trade_date)를 뜻합니다)
                    - 스키마에 없는 테이블/컬럼은 절대 쓰지 마세요.
                    - 결과가 많을 수 있으면 LIMIT 을 붙이세요.
                    - SELECT 절에 가능하면 ticker_name, ticker_code, trade_date 를 포함하세요.
                      (결과를 종목별 JSON으로 묶는 데 필요합니다)
                    - "주가 흐름", "추이", "차트" 질문에는 open_price, high_price, low_price, close_price 네 개를 모두 포함하세요.

                    
                종목명 추출 규칙:
                    - ticker_name 에는 회사 이름만 넣으세요.
                    - "요즘", "최근", "오늘", "어제", "지난주", "이번주", "며칠", "얼마 전" 같은 시간 표현은
                      ticker_name 에 넣지 마세요. 이런 표현은 ORDER BY trade_date DESC 와 LIMIT 으로 처리하세요.
                    - 기간을 숫자로 말하지 않았으면 LIMIT 20 을 쓰세요. ("요즘", "최근" → LIMIT 20)
                    - "주식", "주가", "종목", "차트", "흐름", "추이" 같은 단어도 ticker_name 에서 빼세요.
                    - 영문이 섞인 종목명은 대문자로 쓰세요. (sk하이닉스 → 'SK하이닉스')
                    - 예: "요즘 두산로보틱스 주식 흐름 알려줘"
                          → ticker_name = '두산로보틱스'      (O)
                          → ticker_name = '요즘 두산로보틱스'  (X)
                          
                [종목명 약칭] 질문에 아래 약칭이 나오면 오른쪽 공식 종목명으로 조회하세요.
                    삼전 = 삼성전자          하이닉스 = SK하이닉스      현차 = 현대차
                    삼바 = 삼성바이오로직스   셀트 = 셀트리온           한전 = 한국전력
                    두로 = 두산로보틱스       카뱅 = 카카오뱅크          엘지전자 = LG전자

                자주 하는 질문의 올바른 쿼리 예시:
                
                예시1) "경농 최근 5일 종가 보여줘" (특정 종목의 최근 N일 → 그 종목만 필터 후 정렬+LIMIT)
                SELECT ticker_name, ticker_code, trade_date, open_price, high_price, low_price, close_price
                FROM t_stock_price_data
                WHERE ticker_name = '경농' AND del_yn = false
                ORDER BY trade_date DESC
                LIMIT 5
                
                예시2) "오늘 종가가 가장 높은 종목 5개는?" (오늘/최신일 → MAX(trade_date) 서브쿼리)
                SELECT ticker_name, ticker_code, trade_date, close_price
                FROM t_stock_price_data
                WHERE trade_date = (SELECT MAX(trade_date) FROM t_stock_price_data WHERE del_yn = false)
                  AND del_yn = false
                ORDER BY close_price DESC
                LIMIT 5
                
                예시3) "최신일 기준 거래량 가장 많은 종목 3개" (랭킹도 최신일 필터 먼저)
                SELECT ticker_name, ticker_code, trade_date, volume
                FROM t_stock_price_data
                WHERE trade_date = (SELECT MAX(trade_date) FROM t_stock_price_data WHERE del_yn = false)
                  AND del_yn = false
                ORDER BY volume DESC
                LIMIT 3
                
                예시4) "삼성전자 최근 3일 시가 고가 저가 종가" (특정 종목의 최근 N일은 예시1과 같은 패턴)
                SELECT ticker_name, ticker_code, trade_date, open_price, high_price, low_price, close_price
                FROM t_stock_price_data
                WHERE ticker_name = '삼성전자' AND del_yn = false
                ORDER BY trade_date DESC
                LIMIT 3
                
                예시5) "최근 일주일 거래량 가장 높은 종목은?" (전체 종목의 최근 N일 → 최근 거래일 목록 서브쿼리)
                SELECT ticker_name, ticker_code, trade_date, volume
                FROM t_stock_price_data
                WHERE trade_date IN (
                    SELECT DISTINCT trade_date FROM t_stock_price_data
                    WHERE del_yn = false ORDER BY trade_date DESC LIMIT 7
                ) AND del_yn = false
                ORDER BY volume DESC
                LIMIT 5
                
                예시6) "요즘 두산로보틱스 주식 흐름 알려줘" (시간 표현 "요즘"과 접미어 "주식"은 ticker_name 에서 제외)
                SELECT ticker_name, ticker_code, trade_date, open_price, high_price, low_price, close_price
                FROM t_stock_price_data
                WHERE ticker_name = '두산로보틱스' AND del_yn = false
                ORDER BY trade_date DESC
                LIMIT 20
                
                예시7) "삼전 주가 흐름 어때?" (약칭은 공식 종목명으로 바꿔 조회. 삼전 = 삼성전자)
                SELECT ticker_name, ticker_code, trade_date, open_price, high_price, low_price, close_price
                FROM t_stock_price_data
                WHERE ticker_name = '삼성전자' AND del_yn = false
                ORDER BY trade_date DESC
                LIMIT 20
                
                예시8) "하이닉스 최근 5일 주가 어때?" (앞 회사명을 뗀 약칭도 공식 종목명으로 바꿔 조회)
                SELECT ticker_name, ticker_code, trade_date, open_price, high_price, low_price, close_price
                FROM t_stock_price_data
                WHERE ticker_name = 'SK하이닉스' AND del_yn = false
                ORDER BY trade_date DESC
                LIMIT 5

                {error_block}
                질문: "{question}"
                
                SQL:"""

        response = ollama.chat(
            model=self.MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            options=self.OPTIONS,
        )

        # 코드펜스와 해설 문장을 걷어내고 순수 SELECT 문만 남긴다
        sql = self._extract_sql(response["message"]["content"])

        self.logger.info(f"[LLM 생성 SQL] question={question!r}\n{sql}")
        return sql

    @staticmethod
    def _extract_sql(raw_content):
        """
        LLM 응답에서 실행할 SELECT 문만 추출

        모델이 "설명 없이 SQL만 출력하세요" 지시를 어기고
        SQL 앞뒤에 머리말이나 "### 설명" 같은 해설을 붙이는 경우가 있다.
        이를 그대로 넘기면 세 곳에서 문제가 생긴다.
          1. is_safe_sql : 해설 앞의 세미콜론이 다중 문장으로 판정되어 거부
          2. is_safe_sql : 해설에 섞인 영어 단어(update, create 등)가 금지 키워드로 판정
          3. enforce_limit: 문장 끝 LIMIT 정규식이 빗나가 해설 뒤에 LIMIT이 붙어 문법 오류

        추출 후에도 is_safe_sql 검증은 그대로 수행되므로,
        잘라낸 결과가 SELECT가 아니거나 금지 키워드를 포함하면 실행되지 않는다.

        param raw_content: LLM이 반환한 원본 문자열

        return: str - 추출된 SELECT 문 (추출 실패 시 빈 문자열)
        """
        text_block = (raw_content or "").strip()

        # 1. 마크다운 코드펜스 안의 내용만 사용 (```sql ... ``` 형태)
        fenced = re.search(
            r"```[a-zA-Z]*\s*(.+?)\s*```",
            text_block,
            flags=re.DOTALL,
        )
        if fenced:
            text_block = fenced.group(1).strip()
        else:
            # 펜스가 열린 채 끝난 경우까지 대비해 남은 백틱을 제거
            text_block = re.sub(r"^```[a-zA-Z]*\s*", "", text_block)
            text_block = re.sub(r"\s*```$", "", text_block).strip()

        # 2. SELECT 이전의 머리말 제거 (예: "다음은 요청한 쿼리입니다:")
        #    SELECT 앞의 내용은 버려지므로 실행 대상이 될 수 없다
        select_match = re.search(r"\bselect\b", text_block, flags=re.IGNORECASE)
        if not select_match:
            return ""
        text_block = text_block[select_match.start():]

        # 3. 첫 세미콜론까지가 하나의 문장. 그 뒤에 붙은 해설이나 추가 문장은 버린다
        semicolon_index = text_block.find(";")
        if semicolon_index != -1:
            text_block = text_block[:semicolon_index]

        # 4. 세미콜론이 없으면 해설이 시작되는 줄에서 자른다
        sql_lines = []
        for line in text_block.splitlines():
            if EXPLANATION_LINE_PATTERN.match(line):
                break
            sql_lines.append(line)

        return "\n".join(sql_lines).strip()

    def is_safe_sql(self, sql):
        """
        생성된 SQL이 실행해도 안전한지 검증 (SELECT 전용 방어)

        검사 항목:
        1. SELECT 로 시작하는가
        2. 위험 키워드(INSERT/UPDATE/DELETE/DROP 등)가 없는가
        3. 세미콜론으로 여러 문장을 실행하려 하지 않는가

        param sql: generate_sql이 만든 SQL 문자열(str)

        return: (ok, reason) 튜플
                ok(bool)     - 안전하면 True
                reason(str)  - 거부 사유 (안전하면 빈 문자열)
        """
        if not sql:
            return False, "빈 쿼리"

        lowered = sql.lower()

        # 1. SELECT 로 시작해야 함
        if not lowered.lstrip().startswith("select"):
            return False, "SELECT 문이 아님"

        # 2. 세미콜론 다중 문장 차단 (끝의 세미콜론 하나는 허용)
        #    _extract_sql이 첫 세미콜론까지만 남기므로 정상 흐름에서는 걸리지 않는다.
        #    추출을 거치지 않고 직접 호출되는 경우를 위한 2차 방어선으로 유지한다.
        if ";" in sql.rstrip().rstrip(";"):
            return False, "여러 문장(;) 실행은 허용되지 않음"

        # 3. 위험 키워드 차단 (단어 경계로 검사)
        for kw in FORBIDDEN_KEYWORDS:
            if re.search(rf"\b{kw}\b", lowered):
                return False, f"허용되지 않은 키워드: {kw}"

        return True, ""

    def enforce_limit(self, sql):
        """
        LIMIT 이 없으면 DEFAULT_LIMIT 을 붙이고, 너무 크면 MAX_LIMIT 으로 제한

        param sql: 안전 검증을 통과한 SELECT 쿼리(str)

        return: str - LIMIT 이 보정된 쿼리
        """
        sql = sql.rstrip().rstrip(";")
        m = re.search(r"limit\s+(\d+)\s*$", sql, flags=re.IGNORECASE)
        if m:
            n = int(m.group(1))
            if n > MAX_LIMIT:  # 과도한 LIMIT 은 상한으로 교체
                sql = re.sub(r"limit\s+\d+\s*$", f"LIMIT {MAX_LIMIT}", sql, flags=re.IGNORECASE)
        else:
            sql = f"{sql} LIMIT {DEFAULT_LIMIT}"
        return sql

    def run_query(self, sql):
        """
        검증 통과한 SELECT 쿼리를 실행해 결과를 반환

        param sql: 실행할 SELECT 쿼리(str)

        return: (columns, rows) 튜플
                columns(list[str])       - 컬럼명 목록
                rows(list[tuple])         - 행 데이터
        """
        with self.db.get_postgres_db() as session:
            result = session.execute(text(sql))
            columns = list(result.keys())
            rows = result.fetchall()
        return columns, rows

    @staticmethod
    def _to_plain(value):
        """
        DB 값(Decimal, date 등)을 JSON 직렬화 가능한 파이썬 기본 타입으로 변환

        - Decimal: 소수부가 없으면 int, 있으면 float
        - date/datetime: 'YYYY-MM-DD' 형식 문자열 (trade_date가 date 타입이라 필요)

        param value: DB에서 읽은 셀 값

        return: JSON에 넣을 수 있는 값 (int/float/str/bool/None)
        """
        if isinstance(value, Decimal):
            return int(value) if value == value.to_integral_value() else float(value)
        if isinstance(value, (datetime.date, datetime.datetime)):
            return value.strftime("%Y-%m-%d")
        return value

    def rows_to_json(self, columns, rows):
        """
        조회 결과를 종목별 JSON 구조로 변환 (에이전트 공유용 출력 형식)

        출력 형식:
        [
          {
            "ticker_name": "삼성전자",
            "ticker_code": "005930",
            "period_days": 30,          # 실제 조회된 거래일 수 (price_data 길이)
            "price_data": [
              {"trade_date": ..., "open_price": ..., ...},  # 종목/코드 제외 나머지 컬럼
              ...
            ]
          },
          ...
        ]

        ticker_name 컬럼이 조회에 없으면 그룹핑 없이
        [{"ticker_name": None, "ticker_code": None, ...}] 로 반환한다.
        price_data 안의 각 행은 날짜 오름차순으로 정렬한다.

        param columns: 컬럼명 목록(list[str])
        param rows: 행 데이터(list[tuple])

        return: list[dict] - 종목별로 묶인 JSON 구조
        """
        grouped = {}  # ticker_name → {"ticker_code":..., "price_data":[...]}
        order = []    # 종목 등장 순서 유지

        for row in rows:
            record = {col: self._to_plain(val) for col, val in zip(columns, row)}
            name = record.pop("ticker_name", None)
            code = record.pop("ticker_code", None)

            if name not in grouped:
                grouped[name] = {"ticker_name": name, "ticker_code": code,
                                 "period_days": 0, "price_data": []}
                order.append(name)
            grouped[name]["price_data"].append(record)

        result = [grouped[name] for name in order]

        # price_data를 날짜 오름차순 정렬 + period_days(조회된 거래일 수) 확정
        for item in result:
            if item["price_data"] and "trade_date" in item["price_data"][0]:
                item["price_data"].sort(key=lambda r: r["trade_date"])
            item["period_days"] = len(item["price_data"])
        return result

    def _ask_detail(self, question):
        """
        [내부용] 자연어 질문 → SQL 생성/실행 → 상세 결과 dict 반환

        실행 에러가 나면 에러 메시지를 LLM에게 돌려주고 1회 재생성한다(자가 교정).
        디버깅/대화형 출력에서 SQL과 실패 사유까지 보고 싶을 때 사용.

        param question: 사용자 자연어 질문(str)

        return: dict - {"success", "sql", "data", "message"}
        """
        error_feedback = None
        sql = ""
        for attempt in (1, 2):  # 최초 1회 + 에러 교정 재시도 1회
            try:
                sql = self.generate_sql(question, error_feedback)
            except Exception as e:
                self.logger.error(f"SQL 생성 실패: {e}")
                return {"success": False, "sql": "", "data": [],
                        "message": f"SQL 생성 실패(LLM 호출 오류): {e}"}

            ok, reason = self.is_safe_sql(sql)
            if not ok:
                self.logger.warning(f"안전하지 않은 쿼리 거부: {reason} / {sql}")
                return {"success": False, "sql": sql, "data": [],
                        "message": f"안전하지 않은 쿼리라 실행하지 않음: {reason}"}

            sql = self.enforce_limit(sql)
            self.logger.debug(f"[실행 SQL] (시도 {attempt})\n{sql}")

            try:
                columns, rows = self.run_query(sql)
            except Exception as e:
                self.logger.warning(f"쿼리 실행 실패(시도 {attempt}): {e}")
                if attempt == 1:
                    error_feedback = str(e).splitlines()[0]  # 에러 첫 줄만 LLM에 전달
                    continue
                return {"success": False, "sql": sql, "data": [],
                        "message": f"쿼리 실행 실패: {e}"}

            return {"success": True, "sql": sql,
                    "data": self.rows_to_json(columns, rows), "message": ""}

    def ask(self, question):
        """
        [공개 API] 자연어 질문을 받아 LLM이 SQL(SELECT)을 생성하고,
        안전 검증 후 실행해 주가 데이터를 종목별 JSON으로 반환한다 (Text-to-SQL).

        에이전트(Tool Calling) 등 외부 코드가 호출하는 진입점. 예외를 던지지 않는다.

        처리 흐름:
          질문 → generate_sql(LLM이 SELECT 생성) → is_safe_sql(안전 검증)
               → enforce_limit(LIMIT 보정) → run_query(실행)
               → rows_to_json(종목별 JSON 변환) → 결과 반환
          (실행 에러 시 에러 메시지를 LLM에게 돌려주고 1회 재생성하는 자가 교정 포함)

        안전장치:
          - SELECT 문만 허용 (DROP/DELETE/UPDATE/INSERT 등 차단)
          - 세미콜론 다중 문장 차단
          - LIMIT 없으면 자동으로 추가 (결과 폭주 방지)
          - LLM은 스키마 범위 안에서만 쿼리를 만들도록 프롬프트로 제한

        param question: 사용자 자연어 질문(str)
                        예: "삼성전자 최근 5일 종가", "오늘 거래량 가장 많은 종목 3개"

        return: list[dict] - 종목별 주가 데이터. 실패하거나 결과가 없으면 빈 리스트 []
            [
              {
                "ticker_name": "삼성전자",
                "ticker_code": "005930",
                "period_days": 3,                  # 실제 조회된 거래일 수
                "price_data": [                    # 날짜 오름차순
                  {"trade_date": "2026-07-15", "open_price": 283500,
                   "high_price": 284500, "low_price": 273000, "close_price": 279500},
                  ...
                ]
              },
              ...
            ]
        """
        return self._ask_detail(question)["data"]


def main():
    """
    단독 테스트용 대화형 모드 (import 해서 쓸 때는 실행되지 않음)

    질문을 입력받아 ask() 결과(JSON)만 출력한다.
    생성/실행된 SQL은 화면에 노출하지 않고 DEBUG 로그로만 남는다.
    """
    sys.stdout.reconfigure(encoding="utf-8")  # 출력 한글 깨짐 방지
    sys.stdin.reconfigure(encoding="utf-8")   # 입력 한글 깨짐 방지
    analyzer = StockLLMAnalysis()

    while True:
        try:
            question = input("\n질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료합니다.")
            break
        if not question:
            print("종료합니다.")
            break
        data = analyzer.ask(question)
        if data:
            print(json.dumps(data, ensure_ascii=False, indent=2))
        else:
            print("(조회 결과가 없습니다. 질문을 바꿔서 다시 시도해 보세요.)")


if __name__ == "__main__":
    main()