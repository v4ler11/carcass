from pathlib import Path

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from core.config import carcass_env
from tracing.attributes import Attr
from tracing.conversation import ConversationSpanProcessor
from tracing.exporter import JsonlSpanExporter

_provider: TracerProvider | None = None


def build_provider(
    directory: Path,
    *,
    service_name: str = Attr.SERVICE_NAME,
    environment: str | None = None,
) -> TracerProvider:
    """A provider that writes spans to disk, one JSONL file per day."""
    provider = TracerProvider(
        resource=Resource.create(
            {
                "service.name": service_name,
                "deployment.environment.name": environment or carcass_env(),
            }
        )
    )
    provider.add_span_processor(ConversationSpanProcessor())
    provider.add_span_processor(BatchSpanProcessor(JsonlSpanExporter(directory)))
    return provider


def configure(
    directory: Path,
    *,
    service_name: str = Attr.SERVICE_NAME,
    environment: str | None = None,
) -> TracerProvider:
    """Install the disk provider process-wide. Call once, at startup.

    Repeating the call returns the provider already installed: OpenTelemetry
    keeps the first provider and silently ignores later ones, so building a
    second here would only create an exporter that never receives a span.
    """
    global _provider
    if _provider is not None:
        return _provider

    provider = build_provider(
        directory, service_name=service_name, environment=environment
    )
    trace.set_tracer_provider(provider)
    installed = trace.get_tracer_provider()
    if installed is not provider:
        # Someone installed a provider first; that one cannot be replaced.
        provider.shutdown()
        raise RuntimeError(
            "a TracerProvider is already installed; configure() cannot replace it"
        )

    _provider = provider
    return _provider


def shutdown() -> None:
    """Flush queued spans and stop the exporter thread.

    The global provider stays installed - OpenTelemetry does not allow
    replacing it - so a later `configure()` raises instead of installing an
    exporter that would receive nothing.
    """
    global _provider
    if _provider is not None:
        _provider.shutdown()
        _provider = None
