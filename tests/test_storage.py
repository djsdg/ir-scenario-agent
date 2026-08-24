from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from ir_agent.domain import CreateScenarioRequest, InfluenceFactor
from ir_agent.library import open_scenario_library
from ir_agent.sqlite_library import (
    ConcurrentLibraryWriteError,
    SQLiteScenarioLibrary,
    migrate_json_to_sqlite,
)


class SQLiteStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.source = self.root / "source.json"
        shutil.copyfile(Path("data/scenario_library.json"), self.source)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_json_library_can_be_migrated_and_reopened_as_sqlite(self) -> None:
        target_path = self.root / "library.sqlite3"
        library = migrate_json_to_sqlite(self.source, target_path)

        self.assertIsInstance(library, SQLiteScenarioLibrary)
        self.assertEqual(len(library.list_scenarios()), 6)
        reopened = open_scenario_library(target_path)
        self.assertIsInstance(reopened, SQLiteScenarioLibrary)
        self.assertEqual(len(reopened.list_use_cases()), 8)

    def test_sqlite_writes_are_visible_to_another_library_instance(self) -> None:
        target_path = self.root / "library.sqlite3"
        migrate_json_to_sqlite(self.source, target_path)
        first = open_scenario_library(target_path)
        second = open_scenario_library(target_path)

        first.create(
            CreateScenarioRequest(
                name="SQLite 并发测试场景",
                description="验证 SQLite 后端的事务写入可以被另一个实例读取。",
                category="Scenario",
                actor="测试系统",
                influence_factors=[
                    InfluenceFactor(
                        name="测试节点",
                        kind="environment",
                        dimension="hardware_environment",
                        candidate_values=["节点"],
                        selected_values=["节点"],
                    )
                ],
                owner="test",
                business_goal="验证持久化",
                actions=["写入", "读取"],
                constraints=["仅用于测试"],
                lifecycle="正常服务",
            )
        )

        self.assertTrue(any(item.name == "SQLite 并发测试场景" for item in second.list_scenarios()))

        stale_document = first.document()
        second.create(
            CreateScenarioRequest(
                name="SQLite 第二次写入",
                description="验证旧快照不能静默覆盖其他进程刚刚提交的数据。",
                category="Scenario",
                actor="测试系统",
                influence_factors=[
                    InfluenceFactor(
                        name="测试节点",
                        kind="environment",
                        dimension="hardware_environment",
                        candidate_values=["节点"],
                        selected_values=["节点"],
                    )
                ],
                owner="test",
                business_goal="验证乐观并发控制",
                actions=["读取快照", "尝试写入"],
                constraints=["拒绝过期快照"],
                lifecycle="正常服务",
            )
        )
        with self.assertRaises(ConcurrentLibraryWriteError):
            first._atomic_write(stale_document)


if __name__ == "__main__":
    unittest.main()
