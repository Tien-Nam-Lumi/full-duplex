import glob
import os

files = ["app.py", "tts_player.py", "interruption_detector.py", "turn_collector.py", "smart_turn.py", "fake_asr.py"]
out_path = "/Utilisateurs/tnguye28/full-duplex/full_duplex_poc/code_for_chat_gpt.txt"

print("Writing to:", out_path)
with open(out_path, "w", encoding="utf-8") as out:
    for f in files:
        fpath = os.path.join("/Utilisateurs/tnguye28/full-duplex/full_duplex_poc", f)
        if os.path.exists(fpath):
            out.write(f"\n\n{'='*40}\nFile: {f}\n{'='*40}\n\n")
            with open(fpath, "r", encoding="utf-8") as infile:
                out.write(infile.read())
        else:
            print(f"Warning: {f} not found at {fpath}")

print("Done generating bundle!")
