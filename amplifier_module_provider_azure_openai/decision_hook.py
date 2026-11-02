"""Decision-only Azure accounts through Core's supported hook lifecycle."""

__amplifier_module_type__ = "hook"


class DecisionAccount:
    def __init__(self, config):
        from . import _AzureAccountCleanupError, _resolve_azure_account
        try:
            account = _resolve_azure_account(config)
        except _AzureAccountCleanupError:
            from amplifier_decisions.hooks import UnsafeAccountCleanup
            raise UnsafeAccountCleanup("Azure account construction cleanup failed") from None
        if account is None:
            raise ValueError("Azure decision account requires an endpoint and authentication")
        self.account = account
        self.transports = []

    def decision_backend(self, config, coordinator):
        from .decisions import FoundryDecisions
        transport = FoundryDecisions(self.account, config, coordinator)
        self.transports.append(transport)
        return transport.backend

    async def close(self):
        errors = []
        for transport in self.transports:
            try:
                await transport.close()
            except BaseException as exc:
                errors.append(exc)
        self.transports.clear()
        try:
            await self.account.close()
        except BaseException as exc:
            errors.append(exc)
        if errors:
            raise BaseExceptionGroup("Azure decision account cleanup failed", errors)


async def mount(coordinator, config=None):
    from amplifier_decisions.hooks import mount_account
    return await mount_account(
        coordinator, config, provider_factory=DecisionAccount,
        default_id="azure-decisions", hook_module="hooks-azure-openai-decisions",
    )
