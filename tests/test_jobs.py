"""Ingest job history persists across portal restarts."""

from agentloom_media.ui.jobs import JobManager


def test_completed_job_survives_a_new_manager(tmp_path):
    first = JobManager(tmp_path)
    job = first.create_job("https://youtu.be/abc", model="gpt-5.6-luna")
    job.status = "completed"
    job.stage = "completed"
    job.meta = {"title": "A video"}
    job.artifacts = {"agentloom_proposal": "proposals/p.json"}
    job.log("done")
    first.persist(job)

    second = JobManager(tmp_path)
    listed = second.list_jobs()
    assert len(listed) == 1
    assert listed[0]["job_id"] == job.job_id
    assert listed[0]["status"] == "completed"
    assert listed[0]["meta"]["title"] == "A video"
    reloaded = second.get_job(job.job_id)
    assert reloaded is not None
    assert reloaded.logs[-1]["text"] == "done"


def test_running_job_is_marked_interrupted_on_reload(tmp_path):
    first = JobManager(tmp_path)
    job = first.create_job("https://youtu.be/abc")
    job.status = "running"
    job.stage = "distilling"
    first.persist(job)

    second = JobManager(tmp_path)
    reloaded = second.get_job(job.job_id)
    assert reloaded is not None
    assert reloaded.status == "interrupted"
    assert "Re-submit" in (reloaded.error or "")
    # The interrupted marker is written back, so a third boot does not invent a new story.
    third = JobManager(tmp_path)
    assert third.get_job(job.job_id).status == "interrupted"


def test_cli_proposal_appears_in_job_history(tmp_path):
    proposals = tmp_path / "proposals"
    proposals.mkdir()
    (proposals / "proposal-2026-09-08-dao.json").write_text(
        """{
          "timestamp": "2026-09-08T12:37:28",
          "source": {"url": "https://youtu.be/EtU45PsaL1M", "title": "刀姐", "channel": "DAOJIE", "media_id": "EtU45PsaL1M"},
          "models": {"distillation": "gpt-5.6-luna"}
        }""",
        encoding="utf-8",
    )
    manager = JobManager(tmp_path)
    listed = manager.list_jobs()
    assert len(listed) == 1
    assert listed[0]["status"] == "completed"
    assert listed[0]["meta"]["title"] == "刀姐"
    assert listed[0]["artifacts"]["agentloom_proposal"].endswith("proposal-2026-09-08-dao.json")
    # Second boot does not duplicate the import.
    again = JobManager(tmp_path)
    assert len(again.list_jobs()) == 1


def test_corrupt_job_file_is_skipped(tmp_path):
    folder = tmp_path / "data" / "jobs"
    folder.mkdir(parents=True)
    (folder / "broken.json").write_text("{not json", encoding="utf-8")
    manager = JobManager(tmp_path)
    assert manager.list_jobs() == []
