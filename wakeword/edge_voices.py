"""Fixed split of the 47 English Microsoft Edge neural voices into TRAIN and held-out TEST personas.

Rules:
  * a persona's "...MultilingualNeural" variant stays on the same side as its base voice (same speaker);
  * the voices of Housemate's earlier "hey hermes" fixtures (spike S4) are TEST, because those fixtures are
    part of the held-out set;
  * every locale has at least one TEST voice; the remaining voice(s) of the locale go to TRAIN.

No voice in TRAIN is ever used for evaluation, and no voice in TEST is ever used for training.
"""
from __future__ import annotations

TEST = [
    "en-AU-NatashaNeural",
    "en-CA-ClaraNeural",
    "en-GB-RyanNeural", "en-GB-SoniaNeural", "en-GB-ThomasNeural",
    "en-HK-SamNeural",
    "en-IE-EmilyNeural",
    "en-IN-PrabhatNeural",
    "en-KE-AsiliaNeural",
    "en-NG-AbeoNeural",
    "en-NZ-MollyNeural",
    "en-PH-JamesNeural",
    "en-SG-LunaNeural",
    "en-TZ-ElimuNeural",
    "en-US-AndrewNeural", "en-US-AndrewMultilingualNeural", "en-US-AriaNeural",
    "en-US-BrianNeural", "en-US-BrianMultilingualNeural", "en-US-ChristopherNeural",
    "en-US-EmmaNeural", "en-US-EmmaMultilingualNeural", "en-US-GuyNeural", "en-US-JennyNeural",
    "en-ZA-LeahNeural",
]

TRAIN = [
    "en-AU-WilliamMultilingualNeural",
    "en-CA-LiamNeural",
    "en-GB-LibbyNeural", "en-GB-MaisieNeural",
    "en-HK-YanNeural",
    "en-IE-ConnorNeural",
    "en-IN-NeerjaNeural", "en-IN-NeerjaExpressiveNeural",
    "en-KE-ChilembaNeural",
    "en-NG-EzinneNeural",
    "en-NZ-MitchellNeural",
    "en-PH-RosaNeural",
    "en-SG-WayneNeural",
    "en-TZ-ImaniNeural",
    "en-US-AnaNeural", "en-US-AvaNeural", "en-US-AvaMultilingualNeural", "en-US-EricNeural",
    "en-US-MichelleNeural", "en-US-RogerNeural", "en-US-SteffanNeural",
    "en-ZA-LukeNeural",
]

assert not set(TEST) & set(TRAIN)
