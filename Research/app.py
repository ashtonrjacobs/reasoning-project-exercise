from pathlib import Path
from sacrebleu.metrics import BLEU
import json
from collections import defaultdict
folder = Path(__file__).resolve().parent / "trace_paths"
bleu = BLEU(effective_order=True, tokenize="13a", lowercase=False)


def main():   
    output_path = Path(__file__).resolve().parent / "results.txt"
    direct_performance = defaultdict(list)
    teacher_performance = defaultdict(list)
    self_performance = defaultdict(list)
    synthesized_teacher_performance = defaultdict(list)
    log_lines = []
    direct_bleus = []
    teacher_bleus = []
    self_bleus = []
    synthesized_teacher_bleus = []
    en_es_performance = defaultdict(list)
    es_en_performance = defaultdict(list)
    fr_en_performance = defaultdict(list)
    pairs   = ["en-es", "es-en", "fr-en"]
    methods = ["direct", "teacher", "self", "synth_teacher"]

    method_key_map = {
        "direct_translation": "direct",
        "teacher-CoT-translation": "teacher",
        "self-CoT-translation": "self",
        "teacher-Synthesized-CoT-translation": "synth_teacher",
    }

    combined_performance = defaultdict(
        lambda: {pair: {m: [] for m in methods} for pair in pairs}
    )
    if not folder.exists():
        msg = f"Folder not found: {folder}"
        print(msg)
        try:
            with output_path.open("w", encoding="utf-8") as out_f:
                out_f.write(msg + "\n")
        except Exception:
            pass
        return
    for f in folder.glob("*.jsonl"):
        bleu_scores = []
        method = ""
        missing = 0
        if not f.is_file():
            continue
        with f.open("r", encoding="utf-8", errors="replace") as fh:
            for ln, line in enumerate(fh, 1):
                line = line.rstrip("\n")
                if not line:
                    continue
                obj = json.loads(line)
                refs = []
                trans = None
                
                for key, value in obj.items():
                    if "reference" in key.lower():
                        refs.append(value)
                
                if "direct_translation" in obj:
                    method = "direct_translation"
                    if obj["direct_translation"]:
                        trans = obj["direct_translation"]
                    else:
                        missing += 1
                        continue
                elif "teacher-CoT-translation" in obj:
                    method = "teacher-CoT-translation"
                    if obj["teacher-CoT-translation"]:
                        trans = obj["teacher-CoT-translation"]
                    else:
                        missing += 1
                        continue
                elif "self-CoT-translation" in obj:
                    method = "self-CoT-translation"
                    if obj["self-CoT-translation"]:
                        trans = obj["self-CoT-translation"]
                    else:
                        missing += 1
                        continue
                elif "teacher-Synthesized-CoT-translation" in obj:
                    method = "teacher-Synthesized-CoT-translation"
                    if obj["teacher-Synthesized-CoT-translation"]:
                        trans = obj["teacher-Synthesized-CoT-translation"]
                    else:
                        missing += 1
                        continue
                if trans is None:
                    missing += 1
                    continue
                else:
                    if method == "direct_translation":
                        direct_bleus.append(bleu.sentence_score(trans, refs).score)
                        direct_performance[obj["model"]].append(bleu.sentence_score(trans, refs).score)
                        
                    elif method == "teacher-CoT-translation":
                        teacher_bleus.append(bleu.sentence_score(trans, refs).score)
                        teacher_performance[obj["model"]].append(bleu.sentence_score(trans, refs).score)
                    elif method == "self-CoT-translation":
                        self_bleus.append(bleu.sentence_score(trans, refs).score)
                        self_performance[obj["model"]].append(bleu.sentence_score(trans, refs).score)
                    elif method == "teacher-Synthesized-CoT-translation":
                        synthesized_teacher_bleus.append(bleu.sentence_score(trans, refs).score)
                        synthesized_teacher_performance[obj["model"]].append(bleu.sentence_score(trans, refs).score)

                    if obj["lp"] == "en-es":
                        en_es_performance[obj["model"]].append(bleu.sentence_score(trans, refs).score)
                    elif obj["lp"] == "es-en":
                        es_en_performance[obj["model"]].append(bleu.sentence_score(trans, refs).score)
                    elif obj["lp"] == "fr-en":
                        fr_en_performance[obj["model"]].append(bleu.sentence_score(trans, refs).score)
                    bleu_score = bleu.sentence_score(trans, refs)
                    bleu_scores.append(bleu_score.score)
                    model = obj.get("model")
                    pair  = obj.get("lp")
                    meth  = method_key_map.get(method)  

                    if model in combined_performance and pair in combined_performance[model] and meth in combined_performance[model][pair]:
                        combined_performance[model][pair][meth].append(bleu_score.score)
                    else:
                        if model not in combined_performance:
                            combined_performance[model] = {p: {m: [] for m in methods} for p in pairs}
                        if pair not in combined_performance[model]:
                            combined_performance[model][pair] = {m: [] for m in methods}
                        if meth not in combined_performance[model][pair]:
                            combined_performance[model][pair][meth] = []
                        combined_performance[model][pair][meth].append(bleu_score.score)
            
                
        if bleu_scores:
            avg_bleu = sum(bleu_scores) / len(bleu_scores)
            msg = f"File: {f.name}, Method: {method}, Missing: {missing}, Average BLEU: {avg_bleu}\n"
            print(msg)
            log_lines.append(msg)
        else:
            msg = f"File: {f.name}, Method: {method}, Missing: {missing}, No BLEU scores found\n"
            print(msg)
            log_lines.append(msg)

    avg_direct_bleu = sum(direct_bleus) / len(direct_bleus)
    avg_teacher_bleu = sum(teacher_bleus) / len(teacher_bleus)
    avg_self_bleu = sum(self_bleus) / len(self_bleus)  
    avg_synthesized_teacher_bleu = sum(synthesized_teacher_bleus) / len(synthesized_teacher_bleus)

    mean_direct_performance = {}
    mean_teacher_performance = {}
    mean_self_performance = {}
    mean_synthesized_teacher_performance = {}

    mean_en_es_performance = {}
    mean_es_en_performance = {}
    mean_fr_en_performance = {}

    for model, scores in en_es_performance.items():
        mean_en_es_performance[model] = sum(scores) / len(scores)
    for model, scores in es_en_performance.items():
        mean_es_en_performance[model] = sum(scores) / len(scores)
    for model, scores in fr_en_performance.items():
        mean_fr_en_performance[model] = sum(scores) / len(scores)

    for model, scores in direct_performance.items():
        mean_direct_performance[model] = sum(scores) / len(scores)
    for model, scores in teacher_performance.items():
        mean_teacher_performance[model] = sum(scores) / len(scores)
    for model, scores in self_performance.items():
        mean_self_performance[model] = sum(scores) / len(scores)
    for model, scores in synthesized_teacher_performance.items():
        mean_synthesized_teacher_performance[model] = sum(scores) / len(scores)


    mean_combined_performance = {}
    winners_by_model_pair = {}

    for model, pair_map in combined_performance.items():
        mean_combined_performance[model] = {}
        winners_by_model_pair[model] = {}
        for pair, method_map in pair_map.items():
            mean_combined_performance[model][pair] = {}
            for m, scores in method_map.items():
                mean_combined_performance[model][pair][m] = (
                    sum(scores) / len(scores) if scores else None
                )

            method_means = mean_combined_performance[model][pair]
            valid = [(m, v) for m, v in method_means.items() if v is not None]
            if valid:
                best_m, best_v = max(valid, key=lambda t: t[1])
                winners_by_model_pair[model][pair] = {"method": best_m, "bleu": best_v}
            else:
                winners_by_model_pair[model][pair] = {"method": None, "bleu": None}

    def fmt_winners(winners):
        lines = []
        for model, pair_map in winners.items():
            lines.append(f"{model}:")
            for pair, info in pair_map.items():
                lines.append(f"  {pair}: {info['method']} ({info['bleu']})")
        return "\n".join(lines)

    combined_msg = (
        "Per (model × pair) winners by BLEU:\n" + fmt_winners(winners_by_model_pair) + "\n"
    )
    print(combined_msg)
    log_lines.append(combined_msg)
    summary_msg = (
        f"Total Direct BLEUS: {len(direct_bleus)}, Average Direct BLEU: {avg_direct_bleu}\n"
        f"Total Teacher BLEUS: {len(teacher_bleus)}, Average Teacher BLEU: {avg_teacher_bleu}\n"
        f"Synthesized Teacher BLEU: {avg_synthesized_teacher_bleu}\n"
        f"Total Self BLEUS: {len(self_bleus)}, Average Self BLEU: {avg_self_bleu}\n"
    )
    performance_msg = (
        f"Direct Performance: {mean_direct_performance}\n"
        f"Teacher Performance: {mean_teacher_performance}\n"
        f"Self Performance: {mean_self_performance}\n"
        f"Synthesized Teacher Performance: {mean_synthesized_teacher_performance}\n"
        f"En-Es Performance: {mean_en_es_performance}\n"
        f"Es-En Performance: {mean_es_en_performance}\n"
        f"Fr-En Performance: {mean_fr_en_performance}\n"
    )
    print(summary_msg)
    print(performance_msg)
    log_lines.append(summary_msg)
    log_lines.append(performance_msg)
    try:
        with output_path.open("w", encoding="utf-8") as out_f:
            out_f.write("".join(line if line.endswith("\n") else line + "\n" for line in log_lines))
    except Exception:
        pass


if __name__ == "__main__":
    main()
