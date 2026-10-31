"""Canonical plain-text passages; offsets are Unicode character offsets."""
import hashlib
import json
import unicodedata


def canonical(text):
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n")).strip() + "\n"


def passages(text):
    text = canonical(text)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    records, cursor = [], 0
    for part in text.split("\n\n"):
        body = part.strip()
        if body:
            start = text.index(body, cursor)
            end = start + len(body)
            records.append({"id": f"{digest[:16]}:{start}:{end}", "document_sha256": digest,
                            "start": start, "end": end, "text": body})
            assert text[start:end] == body
            cursor = end
    return records


def main():
    raw = "A full charge takes 3 hours.\r\n\r\nIndoor use only."
    result = passages(raw)
    assert result == passages(raw.replace("\r\n", "\n"))
    assert result != passages(raw.replace("3 hours", "4 hours"))
    print(json.dumps({"count": len(result), "offsets": [[r["start"], r["end"]] for r in result],
                      "stable": True, "changed_document_changes_ids": True}))


if __name__ == "__main__":
    main()
