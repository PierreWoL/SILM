import re

def extract_thing_paths(text, max_nodes=5):
    # normalize arrows
    text = (
        text.replace("→", "->")
            .replace("⇒", "->")
            .replace("➜", "->")
            .replace("-->", "->")
            .replace(" > ", " -> ")
    )

    paths = []

    pieces = re.split(r'[\n\r;；。]+', text)

    for piece in pieces:
        piece = piece.strip()
        if "->" not in piece:
            continue

        lower_piece = piece.lower()

        if "thing" not in lower_piece:
            continue

        starts = [m.start() for m in re.finditer(r'\bthing\b', lower_piece)]

        for i, start in enumerate(starts):
            end = starts[i + 1] if i + 1 < len(starts) else len(piece)
            frag = piece[start:end].strip()

            raw_nodes = frag.split("->")
            nodes = []

            for raw in raw_nodes:
                node = raw.strip()

                node = re.sub(r'^[\s`"“”\'‘’\[\](){},:：-]+', '', node)
                node = re.sub(r'[\s`"“”\'‘’\[\](){},:：.。;；]+$', '', node)

                node = re.split(r'[，,。.;；:：]', node)[0].strip()

                if not node:
                    nodes = []
                    break

                nodes.append(node)

            if not nodes:
                continue

            if nodes[0].lower() != "thing":
                continue

            if len(nodes) > max_nodes:
                continue

            if len(nodes) < 2:
                continue

            nodes[0] = "Thing"

            paths.append(" -> ".join(nodes))

    unique_paths = []
    seen = set()

    for p in paths:
        key = p.lower()
        if key not in seen:
            seen.add(key)
            unique_paths.append(p)

    return "\n".join(unique_paths)