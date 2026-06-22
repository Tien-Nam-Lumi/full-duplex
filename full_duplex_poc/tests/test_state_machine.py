from full_duplex_poc.state_machine import FullDuplexState, FullDuplexStateMachine


def test_state_machine_transitions():
    sm = FullDuplexStateMachine()

    assert sm.state == FullDuplexState.IDLE

    sm.set_state(FullDuplexState.LISTENING)
    assert sm.state == FullDuplexState.LISTENING

    sm.set_state(FullDuplexState.SPEAKING)
    assert sm.is_speaking is True

    sm.set_state(FullDuplexState.INTERRUPTED_LISTENING)
    assert sm.is_listening_after_interrupt is True

