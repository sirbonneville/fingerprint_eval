"""Provider-routing selection for the battery (BYOK pinning vs azure exclusion)."""

from argparse import Namespace

from fingerprint_eval.battery import _provider_routing


def _args(**kw):
    base = {"byok_only": False, "ignore_providers": ""}
    base.update(kw)
    return Namespace(**base)


def test_no_routing_by_default():
    assert _provider_routing("openai/gpt-4o", _args()) is None
    assert _provider_routing("anthropic/claude-sonnet-4", _args()) is None


def test_ignore_providers_excludes_named_slugs():
    assert _provider_routing("openai/gpt-4o", _args(ignore_providers="azure")) == {"ignore": ["azure"]}
    assert _provider_routing(
        "openai/gpt-4o", _args(ignore_providers="azure, bedrock")
    ) == {"ignore": ["azure", "bedrock"]}


def test_byok_only_pins_each_model_to_its_first_party_vendor():
    assert _provider_routing("anthropic/claude-sonnet-4", _args(byok_only=True)) == {"only": ["anthropic"]}
    assert _provider_routing("openai/gpt-4o", _args(byok_only=True)) == {"only": ["openai"]}


def test_byok_only_takes_precedence_over_ignore():
    routing = _provider_routing(
        "anthropic/claude-sonnet-4", _args(byok_only=True, ignore_providers="azure")
    )
    assert routing == {"only": ["anthropic"]}
