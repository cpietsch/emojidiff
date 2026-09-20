# Owned GPU complete tiny smoke

Status: failed at the adapter-output contract after successful GPU work.

The tokenizer and renderer passed, the RTX 4080 completed one forward/backward step,
checkpoint read-back passed, and the 16,179-byte checkpoint was durably recovered with
SHA-256 `77e93414e04707efb0469718426bb2364d5a7b22e575e519124d5c628442348f`.
The run is still classified as failed because the inherited NGC entrypoint printed its
banner before the JSON result and the adapter correctly rejected mixed stdout. A new
run will bypass the image entrypoint explicitly.
