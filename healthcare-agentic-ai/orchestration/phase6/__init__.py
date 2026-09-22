"""Isolated Phase 6 API; Phase 5 exports and behavior remain unchanged."""

__all__ = ['Phase6Orchestrator', 'Phase6WorkflowState', 'Phase6Policy']


def __getattr__(name):
    # Lazy exports avoid a cycle between safety contracts and workflow state.
    if name == 'Phase6Orchestrator':
        from .orchestrator import Phase6Orchestrator
        return Phase6Orchestrator
    if name == 'Phase6WorkflowState':
        from .state import Phase6WorkflowState
        return Phase6WorkflowState
    if name == 'Phase6Policy':
        from .policy import Phase6Policy
        return Phase6Policy
    raise AttributeError(name)
