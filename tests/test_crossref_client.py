from unittest.mock import MagicMock
import pytest
import requests

from revsys.clients.crossref import CrossrefAPI


@pytest.fixture
def sample_crossref_data():
    return {
        "message": {
            "items": [
                {
                    "DOI": "10.1000/testdoi",
                    "title": ["Test Title"],
                    "container-title": ["Journal X"],
                    "issued": {"date-parts": [[2023, 5, 20]]},
                    "author": [{"given": "John", "family": "Doe"}],
                    "abstract": "<p>Sample <b>abstract</b> text.</p>",
                    "language": "en",
                    "type": "journal-article",
                    "license": [{"URL": "http://license"}],
                    "URL": "http://example.com/download.pdf",
                    "is-referenced-by-count": 10
                }
            ]
        }
    }

def test_crossref_process_data_cleaning(sample_crossref_data):
    api = CrossrefAPI(rows=5)
    records = api.process_data(sample_crossref_data)
    assert isinstance(records, list) and len(records) == 1
    rec = records[0]
    # Verifica limpeza de HTML no abstract
    assert rec.get('Abstract') == 'Sample abstract text.'
    # Verifica preenchimento de colunas padronizadas
    assert rec.get('DOI') == '10.1000/testdoi'
    assert rec.get('Title') == 'Test Title'
    assert rec.get('Journal') == 'Journal X'
    # Verifica formato de Authors Year
    assert 'Doe' in rec.get('Authors Year')


def test_crossref_session_usage():
    # 1. Verifica inicialização padrão com requests.Session
    api = CrossrefAPI()
    assert isinstance(api.session, requests.Session)

    # 2. Verifica suporte a passagem de sessão customizada
    mock_session = MagicMock(spec=requests.Session)
    mock_response = MagicMock()
    mock_response.json.return_value = {"message": {"items": []}}
    mock_session.get.return_value = mock_response

    custom_api = CrossrefAPI(session=mock_session)
    assert custom_api.session is mock_session

    # 3. Verifica que fetch_data utiliza session.get
    res = custom_api.fetch_data(query="machine learning", offset=0)
    assert res == {"message": {"items": []}}
    mock_session.get.assert_called_once()
    assert mock_session.get.call_args[0][0] == custom_api.base_url
    assert "query" in mock_session.get.call_args[1]["params"]