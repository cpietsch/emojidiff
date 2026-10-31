"""Write the OmniLottie probe inputs: 30 validation icons, a 618 px render each, one prompt each."""
import json, shutil, sys
from pathlib import Path
import pandas as pd

OUT = Path(sys.argv[1])
STYLE = "A flat OpenMoji-style emoji icon with thick black outlines and flat colors"
ITEMS = [
    ("1F493", "a red beating heart", "The heart pulses, scaling up and down rhythmically."),
    ("1F494", "a broken red heart split down the middle", "The two halves drift apart and snap back together."),
    ("2764", "a red heart", "The heart bounces up and down."),
    ("1F608", "a purple smiling face with two horns", "The face tilts gently from side to side."),
    ("1F923", "a yellow face rolling on the floor laughing with tears", "The face rotates back and forth while laughing."),
    ("1FAE8", "a yellow shaking face", "The face shakes rapidly left and right."),
    ("1F970", "a yellow smiling face surrounded by small red hearts", "The hearts float around the face and pulse."),
    ("1F916", "a grey robot head with antenna", "The robot head bobs up and down and the antenna wiggles."),
    ("1F47E", "a purple pixel alien monster", "The monster hops up and down."),
    ("1F40B", "a blue whale", "The whale swims, bobbing gently up and down."),
    ("1F433", "a blue whale spouting water", "The water spout rises and falls above the whale."),
    ("1F415", "a brown dog standing", "The dog bounces and wags its tail."),
    ("1F426", "a small bird", "The bird flaps its wings and moves up and down."),
    ("1F430", "a white rabbit face with long ears", "The rabbit's ears wiggle and the head nods."),
    ("1F99F", "a mosquito", "The mosquito flies in a small loop."),
    ("1F342", "a fallen orange leaf", "The leaf falls, swaying from side to side."),
    ("1F331", "a green seedling with two leaves", "The seedling grows upward from the ground."),
    ("1F344", "a red mushroom with white spots", "The mushroom squashes and stretches."),
    ("1F4A3", "a black bomb with a lit fuse", "The fuse sparks and the bomb wobbles."),
    ("2699", "a grey gear", "The gear rotates continuously."),
    ("1F50B", "a green battery", "The battery charge level fills up."),
    ("1FAE7", "blue soap bubbles", "The bubbles float upward."),
    ("1F30A", "a blue water wave", "The wave rolls forward and back."),
    ("1F329", "a grey cloud with a yellow lightning bolt", "The lightning bolt flashes on and off."),
    ("1F6DE", "a black wheel with a grey rim", "The wheel spins continuously."),
    ("2708", "a grey airplane", "The airplane flies forward, rising and dipping slightly."),
    ("1FA90", "a ringed planet", "The planet floats up and down and slowly rotates."),
    ("2602", "an open umbrella", "The umbrella sways from side to side."),
    ("26A0", "a yellow warning triangle with an exclamation mark", "The warning sign blinks."),
    ("274C", "a red cross mark", "The cross mark pops in, scaling up from nothing."),
]
d = pd.read_parquet("data/manifests/openmoji-17.0.0-curation-v4-reviewed-primary.parquet").set_index("hexcode")
rows = []
for hexcode, what, motion in ITEMS:
    r = d.loc[hexcode]
    assert r.split == "primary/validation", hexcode
    src = Path("data/raw/openmoji/17.0.0/color/618x618") / f"{hexcode}.png"
    shutil.copy(src, OUT / "images" / f"{hexcode}.png")
    rows.append({"hexcode": hexcode, "annotation": r.annotation,
                 "prompt": f"{STYLE}: {what}. {motion}"})
(OUT / "items.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
print(len(rows))
