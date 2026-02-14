"""
=============================================================================
LLD 09: IN-MEMORY FILE SYSTEM
=============================================================================
Design an in-memory file system with directories and files.
Supports: create, read, write, delete, list, find.

KEY CONCEPTS:
  - Composite pattern (File and Directory share interface)
  - Tree traversal
  - Path parsing
=============================================================================
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol


class FSNode(Protocol):
    name: str
    created_at: datetime
    def size(self) -> int: ...


@dataclass
class File:
    name: str
    content: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)

    def size(self) -> int:
        return len(self.content.encode())

    def write(self, content: str) -> None:
        self.content = content
        self.modified_at = datetime.now()

    def append(self, content: str) -> None:
        self.content += content
        self.modified_at = datetime.now()


@dataclass
class Directory:
    name: str
    children: dict[str, File | Directory] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def size(self) -> int:
        return sum(child.size() for child in self.children.values())

    def list_contents(self) -> list[str]:
        result = []
        for name, node in sorted(self.children.items()):
            prefix = "d " if isinstance(node, Directory) else "f "
            result.append(f"{prefix}{name}")
        return result


class FileSystem:
    def __init__(self):
        self.root = Directory("/")

    def _parse_path(self, path: str) -> list[str]:
        """Split path into components."""
        parts = [p for p in path.strip("/").split("/") if p]
        return parts

    def _navigate(self, path: str, create_dirs: bool = False) -> Directory:
        """Navigate to the parent directory of the given path."""
        parts = self._parse_path(path)
        if not parts:
            return self.root

        current = self.root
        # Navigate to parent (all parts except last)
        for part in parts[:-1]:
            if part not in current.children:
                if create_dirs:
                    current.children[part] = Directory(part)
                else:
                    raise FileNotFoundError(f"Directory not found: {part}")
            node = current.children[part]
            if not isinstance(node, Directory):
                raise NotADirectoryError(f"{part} is not a directory")
            current = node
        return current

    def _get_node(self, path: str) -> File | Directory:
        """Get any node by path."""
        if path == "/":
            return self.root
        parent = self._navigate(path)
        name = self._parse_path(path)[-1]
        if name not in parent.children:
            raise FileNotFoundError(f"Not found: {path}")
        return parent.children[name]

    # --- Public API ---
    def mkdir(self, path: str) -> Directory:
        """Create a directory (and parents if needed)."""
        parts = self._parse_path(path)
        current = self.root
        for part in parts:
            if part not in current.children:
                current.children[part] = Directory(part)
            node = current.children[part]
            if not isinstance(node, Directory):
                raise NotADirectoryError(f"{part} is a file")
            current = node
        return current

    def create_file(self, path: str, content: str = "") -> File:
        """Create a file at the given path."""
        parent = self._navigate(path, create_dirs=True)
        name = self._parse_path(path)[-1]
        if name in parent.children:
            raise FileExistsError(f"Already exists: {path}")
        f = File(name, content)
        parent.children[name] = f
        return f

    def read(self, path: str) -> str:
        node = self._get_node(path)
        if not isinstance(node, File):
            raise IsADirectoryError(f"{path} is a directory")
        return node.content

    def write(self, path: str, content: str) -> None:
        node = self._get_node(path)
        if not isinstance(node, File):
            raise IsADirectoryError(f"{path} is a directory")
        node.write(content)

    def delete(self, path: str) -> None:
        parent = self._navigate(path)
        name = self._parse_path(path)[-1]
        if name not in parent.children:
            raise FileNotFoundError(f"Not found: {path}")
        del parent.children[name]

    def ls(self, path: str = "/") -> list[str]:
        node = self._get_node(path)
        if isinstance(node, Directory):
            return node.list_contents()
        return [f"f {node.name}"]

    def find(self, path: str, name_pattern: str) -> list[str]:
        """Find files/dirs matching a name (simple substring match)."""
        import fnmatch
        results = []

        def search(node: File | Directory, current_path: str):
            if fnmatch.fnmatch(node.name, name_pattern):
                results.append(current_path)
            if isinstance(node, Directory):
                for child_name, child in node.children.items():
                    child_path = f"{current_path}/{child_name}".replace("//", "/")
                    search(child, child_path)

        start = self._get_node(path)
        search(start, path)
        return results

    def tree(self, path: str = "/", prefix: str = "") -> str:
        """Print a tree view."""
        node = self._get_node(path)
        lines = [f"{prefix}{node.name}/"]
        if isinstance(node, Directory):
            items = sorted(node.children.items())
            for i, (name, child) in enumerate(items):
                is_last = i == len(items) - 1
                connector = "└── " if is_last else "├── "
                if isinstance(child, Directory):
                    lines.append(f"{prefix}{connector}{name}/")
                    sub_prefix = prefix + ("    " if is_last else "│   ")
                    for sub_name, sub_child in sorted(child.children.items()):
                        sub_is_last = sub_name == sorted(child.children.keys())[-1]
                        sub_conn = "└── " if sub_is_last else "├── "
                        suffix = "/" if isinstance(sub_child, Directory) else f" ({sub_child.size()}B)"
                        lines.append(f"{sub_prefix}{sub_conn}{sub_name}{suffix}")
                else:
                    lines.append(f"{prefix}{connector}{name} ({child.size()}B)")
        return "\n".join(lines)


# --- Demo ---
if __name__ == "__main__":
    fs = FileSystem()

    # Create structure
    fs.mkdir("/home/reza/projects")
    fs.mkdir("/home/reza/docs")
    fs.create_file("/home/reza/projects/app.py", "print('hello')")
    fs.create_file("/home/reza/projects/test.py", "def test(): pass")
    fs.create_file("/home/reza/docs/readme.md", "# My Project")
    fs.create_file("/home/reza/.bashrc", "export PATH=$PATH")

    # List
    print("ls /home/reza:")
    for item in fs.ls("/home/reza"):
        print(f"  {item}")

    # Read
    print(f"\ncat /home/reza/projects/app.py:")
    print(f"  {fs.read('/home/reza/projects/app.py')}")

    # Write
    fs.write("/home/reza/projects/app.py", "print('updated!')")
    print(f"\nAfter write:")
    print(f"  {fs.read('/home/reza/projects/app.py')}")

    # Find
    print(f"\nfind /home -name '*.py':")
    for path in fs.find("/home", "*.py"):
        print(f"  {path}")

    # Tree
    print(f"\ntree /:")
    print(fs.tree("/"))

    # Delete
    fs.delete("/home/reza/docs/readme.md")
    print(f"\nAfter deleting readme.md:")
    print(f"  ls /home/reza/docs: {fs.ls('/home/reza/docs')}")
