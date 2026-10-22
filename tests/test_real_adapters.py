"""Gerçek adaptör testleri — ağa çıkmadan, httpx.MockTransport ile (§3)."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from adapters.anthropic import ANTHROPIC_VERSION, AnthropicAdapter
from adapters.base import PermanentTargetError, RateLimitPolicy, TransientTargetError
from adapters.factory import TargetOptions, build_adapter
from adapters.http_generic import HTTPGenericAdapter, resolve_path
from adapters.openai_compat import DEFAULT_OLLAMA_BASE_URL, OpenAICompatAdapter


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Geliştiricinin gerçek anahtarları testlere sızmasın."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def mock_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def json_responder(
    payload: dict[str, Any], *, status: int = 200, headers: dict[str, str] | None = None
):
    """Gelen isteği kaydeden ve sabit JSON döndüren bir işleyici."""
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(status, json=payload, headers=headers or {})

    return handler, captured


ANTHROPIC_OK = {
    "model": "claude-opus-5",
    "stop_reason": "end_turn",
    "content": [{"type": "text", "text": "Üzgünüm, bunu yapamam."}],
    "usage": {"input_tokens": 12, "output_tokens": 7},
}

OPENAI_OK = {
    "model": "llama3",
    "choices": [
        {"message": {"role": "assistant", "content": "Yardımcı olamam."}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 9, "completion_tokens": 4},
}


class TestAnthropicAdapter:
    def test_istek_bicimi_ve_basliklar(self):
        handler, captured = json_responder(ANTHROPIC_OK)
        adapter = AnthropicAdapter(api_key="test-key", client=mock_client(handler))

        adapter.send("saldırı promptu", system="Sen bir asistansın.")

        request = captured[0]
        assert str(request.url) == "https://api.anthropic.com/v1/messages"
        assert request.headers["x-api-key"] == "test-key"
        assert request.headers["anthropic-version"] == ANTHROPIC_VERSION

        body = json.loads(request.content)
        assert body["system"] == "Sen bir asistansın."
        assert body["messages"] == [{"role": "user", "content": "saldırı promptu"}]
        assert body["max_tokens"] > 0

    def test_yanit_alanlari_eslenir(self):
        handler, _ = json_responder(ANTHROPIC_OK)
        adapter = AnthropicAdapter(api_key="k", client=mock_client(handler))

        response = adapter.send("prompt")

        assert response.text == "Üzgünüm, bunu yapamam."
        assert response.input_tokens == 12
        assert response.output_tokens == 7
        assert response.total_tokens == 19
        assert response.model == "claude-opus-5"
        assert response.raw == ANTHROPIC_OK

    def test_metin_disi_bloklar_atlanir(self):
        payload = {
            **ANTHROPIC_OK,
            "content": [
                {"type": "thinking", "thinking": "gizli akıl yürütme"},
                {"type": "text", "text": "birinci"},
                {"type": "text", "text": " ikinci"},
            ],
        }
        handler, _ = json_responder(payload)
        adapter = AnthropicAdapter(api_key="k", client=mock_client(handler))

        assert adapter.send("p").text == "birinci ikinci"

    def test_saglayici_reddi_metadataya_tasinir(self):
        """Boş içerikli refusal, analiz katmanının kaçırmaması gereken kesin sinyaldir."""
        payload = {**ANTHROPIC_OK, "stop_reason": "refusal", "content": []}
        handler, _ = json_responder(payload)
        adapter = AnthropicAdapter(api_key="k", client=mock_client(handler))

        response = adapter.send("p")
        assert response.text == ""
        assert response.metadata["stop_reason"] == "refusal"

    def test_bozuk_govde_kalici_hata(self):
        handler, _ = json_responder({"usage": {}})
        adapter = AnthropicAdapter(api_key="k", client=mock_client(handler))

        with pytest.raises(PermanentTargetError, match="content"):
            adapter.send("p")

    def test_anahtar_yoksa_kurulmaz(self):
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            AnthropicAdapter()

    def test_anahtar_ortamdan_okunur(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
        handler, captured = json_responder(ANTHROPIC_OK)
        AnthropicAdapter(client=mock_client(handler)).send("p")

        assert captured[0].headers["x-api-key"] == "env-key"

    def test_cok_turlu_destekler(self):
        handler, _ = json_responder(ANTHROPIC_OK)
        assert AnthropicAdapter(api_key="k", client=mock_client(handler)).supports_multi_turn()


class TestOpenAICompatAdapter:
    def test_ollama_base_url_ve_anahtarsiz_calisir(self):
        """Yerel Ollama anahtar istemez; Authorization başlığı hiç gönderilmemeli."""
        handler, captured = json_responder(OPENAI_OK)
        adapter = OpenAICompatAdapter.for_ollama("llama3", client=mock_client(handler))

        adapter.send("prompt", system="sistem")

        request = captured[0]
        assert str(request.url) == f"{DEFAULT_OLLAMA_BASE_URL}/chat/completions"
        assert "authorization" not in request.headers
        assert adapter.name == "ollama:llama3"

    def test_ozel_base_url_sondaki_egik_cizgiyi_tolere_eder(self):
        handler, captured = json_responder(OPENAI_OK)
        adapter = OpenAICompatAdapter(
            model="qwen2.5",
            base_url="http://192.168.1.50:11434/v1/",
            client=mock_client(handler),
        )
        adapter.send("p")

        assert str(captured[0].url) == "http://192.168.1.50:11434/v1/chat/completions"

    def test_anahtar_varsa_bearer_gonderilir(self):
        handler, captured = json_responder(OPENAI_OK)
        adapter = OpenAICompatAdapter(
            model="gpt-4o-mini", api_key="sk-test", client=mock_client(handler)
        )
        adapter.send("p")

        assert captured[0].headers["authorization"] == "Bearer sk-test"

    def test_mesaj_sirasi_sistem_sonra_kullanici(self):
        handler, captured = json_responder(OPENAI_OK)
        OpenAICompatAdapter(model="llama3", client=mock_client(handler)).send(
            "saldırı", system="sen bir asistansın"
        )

        body = json.loads(captured[0].content)
        assert body["messages"] == [
            {"role": "system", "content": "sen bir asistansın"},
            {"role": "user", "content": "saldırı"},
        ]
        assert body["stream"] is False

    def test_varsayilan_sicaklik_sifir(self):
        """Determinizm (§5) için sıcaklık varsayılanı 0 olmalı."""
        handler, captured = json_responder(OPENAI_OK)
        OpenAICompatAdapter(model="llama3", client=mock_client(handler)).send("p")

        assert json.loads(captured[0].content)["temperature"] == 0.0

    def test_yanit_alanlari_eslenir(self):
        handler, _ = json_responder(OPENAI_OK)
        response = OpenAICompatAdapter(model="llama3", client=mock_client(handler)).send("p")

        assert response.text == "Yardımcı olamam."
        assert response.input_tokens == 9
        assert response.output_tokens == 4
        assert response.metadata["finish_reason"] == "stop"

    def test_null_icerik_bos_metne_cevrilir(self):
        """İçerik filtresi devreye girdiğinde content null gelebilir."""
        payload = {"choices": [{"message": {"content": None}, "finish_reason": "content_filter"}]}
        handler, _ = json_responder(payload)

        assert OpenAICompatAdapter(model="m", client=mock_client(handler)).send("p").text == ""

    def test_bos_choices_kalici_hata(self):
        handler, _ = json_responder({"choices": []})
        adapter = OpenAICompatAdapter(model="m", client=mock_client(handler))

        with pytest.raises(PermanentTargetError, match="choices"):
            adapter.send("p")

    def test_anahtar_zorunlu_kilinabilir(self):
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            OpenAICompatAdapter(model="m", require_api_key=True)

    def test_ollama_oran_siniri_yerel_icin_gevsetilir(self):
        handler, _ = json_responder(OPENAI_OK)
        policy = OpenAICompatAdapter.for_ollama("llama3", client=mock_client(handler)).rate_limit()

        assert policy.requests_per_second > RateLimitPolicy().requests_per_second


class TestHataEslemesi:
    """HTTP durum kodlarının geçici/kalıcı ayrımına eşlenmesi (§3)."""

    @pytest.mark.parametrize("status", [429, 500, 502, 503, 529])
    def test_gecici_durumlar(self, status):
        handler, _ = json_responder({"error": {"message": "dur"}}, status=status)
        adapter = OpenAICompatAdapter(model="m", client=mock_client(handler))

        with pytest.raises(TransientTargetError):
            adapter.send("p")

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
    def test_kalici_durumlar(self, status):
        handler, _ = json_responder({"error": {"message": "olmaz"}}, status=status)
        adapter = OpenAICompatAdapter(model="m", client=mock_client(handler))

        with pytest.raises(PermanentTargetError):
            adapter.send("p")

    def test_retry_after_basligi_okunur(self):
        handler, _ = json_responder({}, status=429, headers={"retry-after": "7"})
        adapter = OpenAICompatAdapter(model="m", client=mock_client(handler))

        with pytest.raises(TransientTargetError) as excinfo:
            adapter.send("p")
        assert excinfo.value.retry_after == 7.0

    def test_bozuk_retry_after_yok_sayilir(self):
        handler, _ = json_responder({}, status=429, headers={"retry-after": "Wed, 21 Oct 2026"})
        adapter = OpenAICompatAdapter(model="m", client=mock_client(handler))

        with pytest.raises(TransientTargetError) as excinfo:
            adapter.send("p")
        assert excinfo.value.retry_after is None

    def test_zaman_asimi_gecicidir(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("çok uzun sürdü", request=request)

        adapter = OpenAICompatAdapter(model="m", client=mock_client(handler))
        with pytest.raises(TransientTargetError, match="zaman aşımı"):
            adapter.send("p")

    def test_baglanti_hatasi_gecicidir(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("bağlantı reddedildi", request=request)

        adapter = OpenAICompatAdapter(model="m", client=mock_client(handler))
        with pytest.raises(TransientTargetError, match="bağlantı"):
            adapter.send("p")

    def test_hata_govdesi_json_degilse_metin_kullanilir(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="Gateway patladı")

        adapter = OpenAICompatAdapter(model="m", client=mock_client(handler))
        with pytest.raises(TransientTargetError, match="Gateway patladı"):
            adapter.send("p")


class TestResolvePath:
    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            ("a.b", "x"),
            ("list[0]", 1),
            ("list[2]", 3),
            ("nested[1].key", "v"),
        ],
    )
    def test_cozumleme(self, path, expected):
        data = {"a": {"b": "x"}, "list": [1, 2, 3], "nested": [{}, {"key": "v"}]}
        assert resolve_path(data, path) == expected

    @pytest.mark.parametrize("path", ["yok", "a.yok", "list[9]"])
    def test_bulunamayan_yol_kalici_hata(self, path):
        with pytest.raises(PermanentTargetError, match="çözümlenemedi"):
            resolve_path({"a": {"b": "x"}, "list": [1]}, path)


class TestHTTPGenericAdapter:
    CONFIG: dict[str, Any] = {
        "name": "musteri-asistani",
        "url": "https://api.ornek.com/chat",
        "body_template": {"mesaj": "{{prompt}}", "sistem": "{{system}}", "kanal": "web"},
        "response_path": "data.reply.text",
    }

    def test_govde_sablonu_doldurulur(self):
        handler, captured = json_responder({"data": {"reply": {"text": "merhaba"}}})
        adapter = HTTPGenericAdapter(**self.CONFIG, client=mock_client(handler))

        response = adapter.send("saldırı", system="sistem promptu")

        assert json.loads(captured[0].content) == {
            "mesaj": "saldırı",
            "sistem": "sistem promptu",
            "kanal": "web",
        }
        assert response.text == "merhaba"

    def test_sistem_yoksa_bos_dizeye_cevrilir(self):
        handler, captured = json_responder({"data": {"reply": {"text": "ok"}}})
        HTTPGenericAdapter(**self.CONFIG, client=mock_client(handler)).send("p")

        assert json.loads(captured[0].content)["sistem"] == ""

    def test_ic_ice_ve_liste_sablonlari(self):
        config = {
            **self.CONFIG,
            "body_template": {"turns": [{"role": "user", "text": "{{prompt}}"}], "n": 1},
        }
        handler, captured = json_responder({"data": {"reply": {"text": "ok"}}})
        HTTPGenericAdapter(**config, client=mock_client(handler)).send("selam")

        body = json.loads(captured[0].content)
        assert body["turns"][0]["text"] == "selam"
        assert body["n"] == 1

    def test_prompt_yer_tutucusu_zorunlu(self):
        config = {**self.CONFIG, "body_template": {"mesaj": "sabit"}}
        with pytest.raises(ValueError, match="{{prompt}}"):
            HTTPGenericAdapter(**config)

    def test_metin_olmayan_yanit_kalici_hata(self):
        handler, _ = json_responder({"data": {"reply": {"text": {"nested": 1}}}})
        adapter = HTTPGenericAdapter(**self.CONFIG, client=mock_client(handler))

        with pytest.raises(PermanentTargetError, match="metin değil"):
            adapter.send("p")

    def test_basliklarda_ortam_degiskeni_genisletilir(self, monkeypatch):
        monkeypatch.setenv("MUSTERI_TOKEN", "gizli-123")
        handler, captured = json_responder({"data": {"reply": {"text": "ok"}}})

        adapter = HTTPGenericAdapter(
            **self.CONFIG,
            headers={"authorization": "Bearer ${MUSTERI_TOKEN}"},
            client=mock_client(handler),
        )
        adapter.send("p")

        assert captured[0].headers["authorization"] == "Bearer gizli-123"

    def test_tanimsiz_ortam_degiskeni_hata(self, monkeypatch):
        monkeypatch.delenv("YOK_BOYLE", raising=False)
        with pytest.raises(ValueError, match="YOK_BOYLE"):
            HTTPGenericAdapter(**self.CONFIG, headers={"x": "${YOK_BOYLE}"})

    def test_cok_turlu_varsayilan_kapali(self):
        adapter = HTTPGenericAdapter(**self.CONFIG)
        assert adapter.supports_multi_turn() is False

    def test_yapilandirma_dosyasindan_kurulur(self, tmp_path):
        import yaml

        path = tmp_path / "hedef.yaml"
        path.write_text(
            yaml.safe_dump({**self.CONFIG, "rate_limit": {"requests_per_second": 3.0}}),
            encoding="utf-8",
        )

        handler, _ = json_responder({"data": {"reply": {"text": "ok"}}})
        adapter = HTTPGenericAdapter.from_config_file(path, client=mock_client(handler))

        assert adapter.name == "musteri-asistani"
        assert adapter.rate_limit().requests_per_second == 3.0

    def test_eksik_alan_anlamli_hata(self, tmp_path):
        import yaml

        path = tmp_path / "eksik.yaml"
        path.write_text(yaml.safe_dump({"name": "x"}), encoding="utf-8")

        with pytest.raises(ValueError, match="zorunlu alan"):
            HTTPGenericAdapter.from_config_file(path)


class TestFactory:
    def test_ollama_kisayolu(self):
        adapter = build_adapter(TargetOptions(kind="ollama", model="llama3"))
        assert adapter.name == "ollama:llama3"
        adapter.close()

    def test_openai_compat_base_url_gecer(self):
        adapter = build_adapter(
            TargetOptions(kind="openai-compat", model="m", base_url="http://localhost:8000/v1")
        )
        assert adapter.name == "openai-compat:m"
        adapter.close()

    def test_model_zorunlu(self):
        with pytest.raises(ValueError, match="--model"):
            build_adapter(TargetOptions(kind="ollama"))

    def test_http_generic_config_zorunlu(self):
        with pytest.raises(ValueError, match="--config"):
            build_adapter(TargetOptions(kind="http-generic"))

    def test_bilinmeyen_tur(self):
        with pytest.raises(ValueError, match="bilinmeyen hedef türü"):
            build_adapter(TargetOptions(kind="uydurma"))

    def test_oran_siniri_gecersiz_kilinabilir(self):
        adapter = build_adapter(
            TargetOptions(kind="ollama", model="m", requests_per_second=9.0, max_concurrency=3)
        )
        assert adapter.rate_limit().requests_per_second == 9.0
        assert adapter.rate_limit().max_concurrency == 3
        adapter.close()

    def test_verilmeyen_secenek_adaptor_varsayilanini_ezmez(self):
        adapter = build_adapter(TargetOptions(kind="ollama", model="m"))
        assert adapter.rate_limit() == OpenAICompatAdapter.for_ollama("m").rate_limit()
        adapter.close()
