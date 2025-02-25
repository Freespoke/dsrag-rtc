import os
from time import perf_counter
from dsrag.dsparse.main import parse_and_chunk
from transcript import transcript

os.environ["OPENAI_API_KEY"] = ""
# This frequently reports errors when run on Python 3.12.9
# start errors are more common but we also see contiguity errors in our database
def main():
    for _ in range(0, 10):
        start_time = perf_counter()
        sections, chunks = parse_and_chunk(
            kb_id="test",
            doc_id=6993,
            text=transcript,
            semantic_sectioning_config={
                "use_semantic_sectioning": True,
                "llm_provider": "openai",
                "model": "gpt-4o-mini",
                "language": "english",
            },
        )
        end_time = perf_counter()
        contiguity_errors = check_contiguity(sections)
        bad_start_errors = check_start_idx(sections)
        word_count_errors = check_word_counts(sections, transcript)
        section_errors = check_segment_offsets(sections, transcript)
        if contiguity_errors + bad_start_errors + word_count_errors + section_errors == 0:
            print("OK")
        else:
            print(f"{contiguity_errors} contiguity errors")
            print(f"{bad_start_errors} start_errors")
            print(f"{word_count_errors} word count errors")
        print(f"Segmented in {end_time - start_time:.1f}")

def check_contiguity(sections) -> int:
    errcount = 0
    prev_end = -1
    for section in sections:
        # last section ended at 8, next should start at 9
        if section["start"] != prev_end + 1:
            print(f"{section["start"]} did not start immediately after previous section end {prev_end}. Section: {section}")
            errcount += 1
        prev_end = section["end"]
    return errcount

def check_start_idx(sections):
    errcount = 0
    for section in sections:
        if section["start"] > section["end"]:
            print(f"{section["start"]} > {section["end"]}. Section: {section}")
            errcount += 1
    return errcount

def check_word_counts(sections, transcript):
    errcount = 0
    transcript_wc = len(transcript.split())
    sections_wc = sum([section["word_count"] for section in sections])
    if transcript_wc != sections_wc:
        print(f"Word count mismatch. Expected {transcript_wc}, got {sections_wc}.")
        errcount += 1
    return errcount

def check_segment_offsets(sections, transcript):
    """
    Compare each section with the corresponding lines from the transcript
    using the start and end indices of the section.
    If the text doesn't match or the indices are out of range, print an error.
    Return the number of errors found.
    """
    utterances = transcript.split('\n')
    errcount = 0
    for section in sections:
        chunk = "\n".join(utterances[section["start"] : section["end"] + 1])
        try:
            content =section.get("content", "")
            if content != chunk:
                print(f"Section content does not match corresponding utterances (start: {section["start"]}, end: {section["end"]}). Section: {section}")
                errcount += 1
        except IndexError:
            errcount += 1
            print(f"Section start/end out of range. start {section["start"]}; end {section["end"]}; max {len(utterances)}. Section {section}")
    return errcount

if __name__ == "__main__":
    main()
