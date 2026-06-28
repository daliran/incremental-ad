from pathlib import Path


def git_commit() -> str | None:
    """Return the current HEAD commit hash by reading .git/HEAD directly."""

    try:
        # Walk up from this file to find the .git directory
        search = Path(__file__).resolve().parent
        while search != search.parent:
            head = search / ".git" / "HEAD"
            if head.exists():
                content = head.read_text().strip()
                if content.startswith("ref: "):
                    # Normal branch: resolve the ref to its commit hash
                    ref_file = search / ".git" / content[5:]
                    return ref_file.read_text().strip() if ref_file.exists() else None
                # Detached HEAD: content is the commit hash directly
                return content
            search = search.parent
        return None
    except Exception:
        return None
