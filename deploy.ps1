Write-Host "Bundling Phase 1 RL Training Stack for Nebius AI Cloud..." -ForegroundColor Cyan

# Remove any old archives just in case
if (Test-Path "g1_training_stack.tar.gz") {
    Remove-Item "g1_training_stack.tar.gz" -Force
}

# Compress the directory excluding virtual environments and logs
Write-Host "Compressing models, scripts, and requirements into g1_training_stack.tar.gz..."
tar --exclude=".venv" --exclude="logs" --exclude="checkpoints" --exclude="__pycache__" --exclude="mock_sim_env.py" --exclude="*.zip" -czvf g1_training_stack.tar.gz models sim_env.py train_policy.py requirements.txt

Write-Host "`nArchive created successfully!" -ForegroundColor Green

Write-Host "`n========================================================" -ForegroundColor Yellow
Write-Host "              NEBIUS CLOUD DEPLOYMENT GUIDE             " -ForegroundColor Yellow
Write-Host "========================================================" -ForegroundColor Yellow

Write-Host "`nStep 1: Transfer the payload to your Nebius node" -ForegroundColor White
Write-Host "Run the following SCP command from this terminal session:" -ForegroundColor Gray
Write-Host "scp .\g1_training_stack.tar.gz <REMOTE_USER>@<NEBIUS_IP>:~/" -ForegroundColor Cyan

Write-Host "`nStep 2: Connect to your Nebius instance" -ForegroundColor White
Write-Host "ssh <REMOTE_USER>@<NEBIUS_IP>" -ForegroundColor Cyan

Write-Host "`nStep 3: Run the following setup and execution script on Linux" -ForegroundColor White
Write-Host @"
# Extract the payload
mkdir -p g1_training_stack
mv g1_training_stack.tar.gz g1_training_stack/
cd g1_training_stack
tar -xzvf g1_training_stack.tar.gz

# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
source `$HOME/.cargo/env

# Initialize Python 3.11 environment
uv venv --python 3.11
source .venv/bin/activate

# Install RL dependencies (this uses pre-compiled Linux wheels)
uv pip install -r requirements.txt

# Launch parallel headless training loop in the background
nohup python train_policy.py --num_envs 16 --total_timesteps 2000000 > training_output.log 2>&1 &
"@ -ForegroundColor Cyan

Write-Host "`nStep 4: Monitor Training" -ForegroundColor White
Write-Host "You can monitor the output with: tail -f training_output.log" -ForegroundColor Gray
Write-Host "Or forward the tensorboard ports to view the reward curves: tensorboard --logdir=./logs" -ForegroundColor Gray
Write-Host "========================================================" -ForegroundColor Yellow
