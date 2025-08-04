module.exports = {
  apps: [
    {
      name: "mining_alerts",
      script: "./start_mining.sh",
      cwd: "/home/fullzecure/dev/google/gemma3n/vlm_alerts",
      watch: true,
      autorestart: true,
      max_memory_restart: "1G",
      env: {
        NODE_ENV: "production"
      }
    },
   {
      name: "gemma_alerts",
      script: "./start_gemma.sh",
      cwd: "/home/fullzecure/dev/google/gemma3n",
      watch: true,
      autorestart: true,
      max_memory_restart: "1G",
    }
  ]
};
