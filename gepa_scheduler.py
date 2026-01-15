from apscheduler.schedulers.blocking import BlockingScheduler
from optimize_gepa import run_gepa_optimization

scheduler = BlockingScheduler(timezone="Asia/Kolkata")

scheduler.add_job(
    run_gepa_optimization,
    trigger="interval",
    hours=24,
    id="gepa_daily_job",
    max_instances=1,
    coalesce=True,
)

if __name__ == "__main__":
    print("GEPA scheduler started")
    scheduler.start()
