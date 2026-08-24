from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .library import tokenize


@dataclass(frozen=True, slots=True)
class Skill:
    """An application-level skill loaded from a trusted Markdown file."""

    name: str
    description: str
    instructions: str
    tags: tuple[str, ...]
    path: Path

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "tags": list(self.tags),
            "path": str(self.path),
        }


@dataclass(frozen=True, slots=True)
class SkillMatch:
    skill: Skill
    score: float
    matched_terms: tuple[str, ...]


class SkillCatalog:
    """Discover and select Markdown skills without adding a YAML dependency."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _files(self) -> list[Path]:
        if not self.root.exists():
            return []
        files = list(self.root.rglob("SKILL.md"))
        files.extend(path for path in self.root.glob("*.md") if path.name != "SKILL.md")
        return sorted(set(files))

    def _read(self, path: Path) -> Skill:
        text = path.read_text(encoding="utf-8")
        metadata, instructions = _split_front_matter(text)
        fallback_name = path.parent.name if path.name.casefold() == "skill.md" else path.stem
        name = str(metadata.get("name") or fallback_name).strip()
        description = str(metadata.get("description") or "").strip()
        raw_tags = metadata.get("tags", [])
        if isinstance(raw_tags, str):
            raw_tags = [raw_tags]
        tags = tuple(str(item).strip() for item in raw_tags if str(item).strip())
        if not description:
            description = instructions.splitlines()[0][:240] if instructions else name
        return Skill(
            name=name,
            description=description,
            instructions=instructions.strip(),
            tags=tags,
            path=path,
        )

    def list(self) -> list[Skill]:
        return [self._read(path) for path in self._files()]

    def get(self, name: str) -> Skill:
        normalized = name.casefold()
        for skill in self.list():
            if skill.name.casefold() == normalized or skill.path.stem.casefold() == normalized:
                return skill
        raise KeyError(f"Unknown skill: {name}")

    def search(self, query: str, *, top_k: int = 5) -> list[SkillMatch]:
        if not query.strip():
            return []
        query_terms = set(tokenize(query))
        if not query_terms:
            return []

        results: list[SkillMatch] = []
        for skill in self.list():
            document_terms = set(
                tokenize(" ".join([skill.name, skill.description, *skill.tags]))
            )
            matched = query_terms & document_terms
            if not matched:
                continue
            score = round(len(matched) / len(query_terms), 4)
            results.append(
                SkillMatch(skill=skill, score=score, matched_terms=tuple(sorted(matched)))
            )
        results.sort(key=lambda item: (-item.score, item.skill.name))
        return results[:top_k]

    def prompt_context(self, query: str, *, top_k: int = 2, min_score: float = 0.2) -> str:
        selected = [item for item in self.search(query, top_k=top_k) if item.score >= min_score]
        if not selected:
            return ""
        blocks = [
            "<active_skills>",
            "以下是根据当前需求自动选中的项目 Skill。它们是项目维护者提供的工作流约束：",
        ]
        for item in selected:
            blocks.extend(
                [
                    f"<skill name={json.dumps(item.skill.name, ensure_ascii=False)}>",
                    item.skill.instructions[:8_000],
                    "</skill>",
                ]
            )
        blocks.append("</active_skills>")
        return "\n" + "\n".join(blocks)


def _split_front_matter(text: str) -> tuple[dict[str, object], str]:
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    try:
        closing_index = next(index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---")
    except StopIteration:
        return {}, text

    metadata: dict[str, object] = {}
    for line in lines[1:closing_index]:
        key, separator, value = line.partition(":")
        if not separator:
            continue
        key = key.strip()
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                parsed = [part.strip() for part in value[1:-1].split(",")]
            metadata[key] = parsed if isinstance(parsed, list) else [value]
        else:
            metadata[key] = value.strip('"\'')
    return metadata, "\n".join(lines[closing_index + 1 :])
