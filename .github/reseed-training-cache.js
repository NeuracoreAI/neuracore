const {saveCache} = require("@actions/cache");
const fs = require("fs");

// Job IDs from start-training run 164 attempt 2. Attempt 1 already reserved
// training-job-<algorithm>-staging-164, so these are stored under a new key
// with the same prefix the evaluate workflow restores.
const jobs = [
  ["ACT", "9d28a8ee-a45e-407e-972b-e1cbc6809359"],
  ["CNNMLP", "04d3ac9c-cfe4-4c7b-a361-3b42db15a497"],
  ["DiffusionPolicy", "518ff560-d1e8-41d2-a970-734d90e01d0c"],
  ["Groot", "0a2df1f3-1e0a-4d6b-b59b-bd9c4a93e249"],
  ["Pi0", "0bfbe081-a166-4246-addb-ad171b994173"],
  ["Pi05", "6d4c77c0-22aa-4d34-9096-e8f608b1dd81"],
];

async function main() {
  const workspace = process.env.GITHUB_WORKSPACE;
  if (!workspace) {
    throw new Error("GITHUB_WORKSPACE is unset");
  }
  process.chdir(workspace);
  console.log(
    `cache service: results=${Boolean(process.env.ACTIONS_RESULTS_URL)} cache=${Boolean(process.env.ACTIONS_CACHE_URL)} token=${Boolean(process.env.ACTIONS_RUNTIME_TOKEN)}`
  );
  if (!process.env.ACTIONS_RUNTIME_TOKEN) {
    throw new Error("ACTIONS_RUNTIME_TOKEN is unset");
  }

  for (const [algorithm, jobId] of jobs) {
    fs.writeFileSync("training_job_id.txt", `${jobId}\n`);
    fs.writeFileSync("test_harness_ref.txt", "main\n");
    const key = `training-job-${algorithm}-staging-164-attempt2`;
    const cacheId = await saveCache(
      ["training_job_id.txt", "test_harness_ref.txt"],
      key
    );
    console.log(`saved ${key} cacheId=${cacheId} job=${jobId}`);
    if (cacheId === -1) {
      throw new Error(`failed to reserve ${key}`);
    }
  }
}

main().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
