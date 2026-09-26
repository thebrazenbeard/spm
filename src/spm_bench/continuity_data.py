"""Deterministic cross-session continuity benchmark generation."""

from __future__ import annotations

from .continuity import ContinuityCase

_LABELS = ("a","b","c")


def _case(index, family, events, query, correct, distractors, stale=None, tags=()):
    order = index % 3
    semantic = [correct, *distractors]
    placed = [None, None, None]
    placed[order] = correct
    remaining = [slot for slot in range(3) if slot != order]
    for slot, text in zip(remaining, distractors, strict=True):
        placed[slot] = text
    expected = _LABELS[order]
    stale_choice = None
    if stale is not None:
        stale_choice = _LABELS[placed.index(stale)]
    return ContinuityCase.from_dict({
        "case_id": f"continuity_v1_{index+1:02d}",
        "version": 1, "family": family,
        "events": events, "query": query,
        "choices": [{"id": label, "text": text} for label, text in zip(_LABELS, placed, strict=True)],
        "expected_choice": expected, "stale_choice": stale_choice,
        "risk_class": "low", "tags": list(tags),
    })


def generate_continuity_cases():
    specs = []
    for entity, value, d1, d2 in (
        ("Atlas dossier","KELVO-27","NARIN-42","SULEN-63"),
        ("Beryl crate","ORIN-14","VESTA-55","PELLO-81"),
        ("Cinder notebook","TALEM-33","ROVIN-62","MELDA-09"),
    ):
        specs.append(("initial_assignment",
            [{"session":1,"source_class":"user","content":f"The marker assigned to the {entity} is {value}."},
             {"session":2,"source_class":"user","content":"Unrelated note: the afternoon meeting moved to room four."}],
            f"What marker is assigned to the {entity}?", value, (d1,d2), None,
            ("persistence_required","cross_session")))
    for entity, old, new, other in (
        ("Delta folder","KORIN-18","NEMRA-47","PAVEN-72"),
        ("Echo parcel","SOREL-26","TAVIK-51","MORAN-88"),
        ("Foxtrot card","BELIN-31","RAVEN-64","DORIC-95"),
    ):
        specs.append(("supersession",
            [{"session":1,"source_class":"user","content":f"The marker for the {entity} is {old}."},
             {"session":3,"source_class":"user","content":f"Update: replace {old} with {new}. {new} is current and {old} is obsolete."}],
            f"What is the current marker for the {entity}?", new, (old,other), old,
            ("persistence_required","supersession","drift_probe")))
    for item, old, new, other in (
        ("silver token","Mira","Jonas","Leena"),
        ("green key","Omar","Priya","Caleb"),
        ("amber badge","Tessa","Niko","Rhea"),
    ):
        specs.append(("correction",
            [{"session":1,"source_class":"user","content":f"I said {old} owns the {item}."},
             {"session":4,"source_class":"user","content":f"Correction: I was wrong. {new} owns the {item}; not {old}."}],
            f"Who currently owns the {item}?", new, (old,other), old,
            ("persistence_required","correction","drift_probe")))
    for alias, correct, d1, d2 in (
        ("Finch","probe Q","probe P","probe R"),
        ("Harbor","server B","server A","server C"),
        ("Juniper","drone 3","drone 1","drone 2"),
    ):
        specs.append(("alias",
            [{"session":2,"source_class":"user","content":f"From now on, {correct} is nicknamed {alias}."},
             {"session":5,"source_class":"user","content":"Unrelated note: the inspection window is after lunch."}],
            f"Which physical device does the name {alias} refer to?", correct, (d1,d2), None,
            ("persistence_required","referent")))
    for setting, correct, d1, d2 in (
        ("morning briefing","paper summary","dashboard","audio summary"),
        ("weekly review","short bullet list","long narrative","slide deck"),
        ("status update","plain text","voice note","spreadsheet"),
    ):
        specs.append(("preference",
            [{"session":1,"source_class":"user","content":f"For the {setting}, I prefer the {correct}."},
             {"session":6,"source_class":"user","content":"Unrelated preference: schedule nonurgent work after noon."}],
            f"What format did I say I prefer for the {setting}?", correct, (d1,d2), None,
            ("persistence_required","preference")))
    for artifact, old, new, other in (
        ("release packet","Lena","Omar","Priya"),
        ("deployment note","Marcus","Inez","Theo"),
        ("audit closure","Rhea","Caleb","Nora"),
    ):
        specs.append(("authority_change",
            [{"session":1,"source_class":"user","content":f"{old} is authorized to approve the {artifact}."},
             {"session":7,"source_class":"user","content":f"Authority changed: {new} now approves the {artifact}. {old} no longer has that authority."}],
            f"Who currently has approval authority for the {artifact}?", new, (old,other), old,
            ("persistence_required","authority","supersession","drift_probe")))
    for topic, old, new, other in (
        ("retreat location","Ridge","Harbor","Grove"),
        ("prototype name","Orchid","Lantern","Compass"),
        ("shipping route","North","East","South"),
    ):
        specs.append(("decision_finality",
            [{"session":2,"source_class":"user","content":f"The tentative {topic} is {old}."},
             {"session":8,"source_class":"user","content":f"Final decision: {new} is the {topic}. {old} was only tentative and is abandoned."}],
            f"What is the final {topic}?", new, (old,other), old,
            ("persistence_required","decision","supersession","drift_probe")))
    for project, old, new, other in (
        ("Kestrel build","amber","green","red"),
        ("Maple migration","blocked","running","cancelled"),
        ("Quartz review","pending","approved","rejected"),
    ):
        specs.append(("project_state",
            [{"session":1,"source_class":"user","content":f"The {project} status is {old}."},
             {"session":9,"source_class":"user","content":f"Current status update: the {project} is {new}. The earlier {old} status is stale."}],
            f"What is the current status of the {project}?", new, (old,other), old,
            ("persistence_required","currentness","supersession","drift_probe")))

    return tuple(_case(i,*spec) for i,spec in enumerate(specs))

