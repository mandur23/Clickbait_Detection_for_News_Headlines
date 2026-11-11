from clickbait.text import clean_text, clean_title, dedup_key


def test_clean_text_strips_markup_entities_and_extra_space() -> None:
    assert clean_text("<b>속보</b>&nbsp;정부,   예산안\n발표") == "속보 정부, 예산안 발표"


def test_clean_text_keeps_clickbait_cues_but_caps_repeats() -> None:
    assert clean_text("결국 터졌다!!!!!! 정체는??") == "결국 터졌다!!! 정체는??"


def test_clean_text_normalizes_quotes_ellipsis_and_fullwidth() -> None:
    assert clean_text("“설마” ‘진짜’…ＡＩ") == "\"설마\" '진짜'...AI"


def test_clean_text_removes_control_and_zero_width_characters() -> None:
    assert clean_text("제목​에\u0007 숨은 문자") == "제목에 숨은 문자"


def test_clean_title_is_clean_text() -> None:
    raw = " <i>정부</i>, 예산안 발표…  "
    assert clean_title(raw) == clean_text(raw)


def test_dedup_key_ignores_spacing_and_case_only() -> None:
    assert dedup_key("AI 반도체  수출 증가") == dedup_key("ai반도체 수출증가")
    assert dedup_key("수출 증가") != dedup_key("수출 감소")
