def test_install_patches_create_agent_and_compile(monkeypatch):
    import jevrail.auto as auto

    auto.uninstall()
    calls = {}

    class Agents:
        @staticmethod
        def create_agent(*args, **kwargs):
            calls["middleware"] = kwargs.get("middleware")
            return "agent"

    class Compiled:
        pass

    class Builder:
        channels = {"messages": object()}

        def compile(self, *args, **kwargs):
            return Compiled()

    monkeypatch.setattr(auto, "_load_agents", lambda: Agents)
    monkeypatch.setattr(auto, "_load_state_graph", lambda: Builder)
    try:
        auto.install(policy=None)
        Agents.create_agent("model", middleware=[])
        assert any(type(item).__name__ == "JevRail" for item in calls["middleware"])
        wrapped = Builder().compile()
        assert type(wrapped).__name__ == "GuardedGraph"
    finally:
        auto.uninstall()
