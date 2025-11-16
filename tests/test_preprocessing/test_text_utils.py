from src.preprocessing.utils.text_utils import clean_wine_name, normalize_string


def test_normalize_string_basic():
    assert normalize_string(" Château-du-Test !! ") == "château du test"
    assert normalize_string(None) == ""
    assert normalize_string("Hello--World") == "hello world"


def test_clean_wine_name_removes_year():
    assert clean_wine_name("Wine 2019") == "wine"
    assert clean_wine_name("1995 Château-Test") == "château test"
    assert clean_wine_name("") == ""
