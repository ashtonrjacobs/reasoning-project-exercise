from pathlib import Path
from sacrebleu.metrics import BLEU
import json

folder = Path(__file__).resolve().parent / "trace_paths"
bleu = BLEU(effective_order=True, tokenize="13a", lowercase=False)


def main():
    bleu_scores = []
    direct_bleus = []
    teacher_bleus = []
    self_bleus = []
    if not folder.exists():
        print(f"Folder not found: {folder}")
        return
    for f in folder.glob("*.jsonl"):
        method = ""
        missing = 0
        if not f.is_file():
            continue
        with f.open("r", encoding="utf-8", errors="replace") as fh:
            for ln, line in enumerate(fh, 1):
                line = line.rstrip("\n")
                if not line:
                    continue
                # raw string
                obj = json.loads(line)
                refs = []
                
                
                for key, value in obj.items():
                    if "reference" in key.lower():
                        refs.append(value)
                
                if "direct_translation" in obj and obj["direct_translation"]:
                    method = "direct_translation"
                    trans = obj["direct_translation"]
                elif "teacher-CoT-translation" in obj and obj["teacher-CoT-translation"]:
                    method = "teacher-CoT-translation"
                    trans = obj["teacher-CoT-translation"]
                elif "self-CoT-translation" in obj and obj["self-CoT-translation"]:
                    trans = obj["self-CoT-translation"]
                    method = "self-CoT-translation"
                else:
                    missing += 1
                    continue
                if method == "direct_translation":
                    direct_bleus.append(bleu.sentence_score(trans, refs).score)
                elif method == "teacher-CoT-translation":
                    teacher_bleus.append(bleu.sentence_score(trans, refs).score)
                elif method == "self-CoT-translation":
                    self_bleus.append(bleu.sentence_score(trans, refs).score)
                bleu_score = bleu.sentence_score(trans, refs)
                bleu_scores.append(bleu_score.score)
        if bleu_scores:
            avg_bleu = sum(bleu_scores) / len(bleu_scores)
            print(f"File: {f.name}, Method: {method}, Missing: {missing}, Average BLEU: {avg_bleu}\n")
        else:
            print(f"File: {f.name}, Method: {method}, Missing: {missing}, No BLEU scores found\n")
    avg_direct_bleu = sum(direct_bleus) / len(direct_bleus)
    avg_teacher_bleu = sum(teacher_bleus) / len(teacher_bleus)
    avg_self_bleu = sum(self_bleus) / len(self_bleus)       
    print(f"Total Direct BLEUS: {len(direct_bleus)}, Average Direct BLEU: {avg_direct_bleu}, Total Teacher BLEUS: {len(teacher_bleus)}, Average Teacher BLEU: {avg_teacher_bleu}, Total Self BLEUS: {len(self_bleus)}, Average Self BLEU: {avg_self_bleu}\n")

if __name__ == "__main__":
    main()