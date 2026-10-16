"""The overrides of the web app's functions in a database's custom_functions."""

import pytest


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """The web app on a database root with one database, "custom", whose custom_functions override get_sorted_kwic."""
    pytest.importorskip("falcon")
    from tests.fixtures.web_app import web_app_client

    root = tmp_path_factory.mktemp("custom_functions")
    (root / "custom" / "data").mkdir(parents=True)
    (root / "custom" / "data" / "db.locals.py").write_text("metadata_sql_types = {}\nmetadata_fields = []\n")
    (root / "custom" / "data" / "web_config.cfg").write_text("access_control = False\n")
    (root / "custom" / "custom_functions").mkdir()
    (root / "custom" / "custom_functions" / "__init__.py").write_text(
        "def get_sorted_kwic(request, config):\n    return {'sorted_by': request.first_kwic_sorting_option}\n"
    )
    with web_app_client(root) as client:
        yield client


@pytest.mark.unit
def test_sorted_kwic(client):
    """The get_sorted_kwic of a database's custom_functions answers for its sorted KWIC, as the overrides of the other
    scripts do for theirs."""
    params = {"q": "love", "first_kwic_sorting_option": "title"}
    response = client.simulate_get("/philologic5/custom/scripts/get_sorted_kwic.py", params=params)
    assert response.status_code == 200, response.text[:300]
    assert response.json == {"sorted_by": "title"}
