from policyflux.toolbox.special_actors.speaker import SequentialSpeaker


def test_speaker_defaults_generate_id_name_and_value() -> None:
    speaker = SequentialSpeaker()

    assert isinstance(speaker.id, int)
    assert speaker.name.startswith("Speaker_")
    assert speaker.agenda_support == 0.5


def test_speaker_constructor_clamps_and_custom_name() -> None:
    speaker = SequentialSpeaker(id=3, name="SpeakerX", agenda_support=2.0)

    assert speaker.id == 3
    assert speaker.name == "SpeakerX"
    assert speaker.agenda_support == 1.0


def test_speaker_get_influence_and_setter_clamp() -> None:
    speaker = SequentialSpeaker(id=4, agenda_support=0.62)

    assert speaker.get_influence() == 0.62

    speaker.set_agenda_support(-1.0)
    assert speaker.agenda_support == 0.0

    speaker.set_agenda_support(5.0)
    assert speaker.agenda_support == 1.0
