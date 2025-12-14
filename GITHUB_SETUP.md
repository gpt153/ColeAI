# GitHub Repository Setup

## Option 1: Using GitHub CLI (Recommended)

If you have `gh` CLI installed:

```bash
# Install gh CLI first (if not installed)
# Ubuntu/Debian: sudo apt install gh
# Mac: brew install gh
# Or: https://cli.github.com/

# Authenticate
gh auth login

# Create repository
gh repo create persona-agent-system --public --description "Multi-persona knowledge bank with PydanticAI and RAG - Create AI agents from any expert's public content"

# Set remote and push
git remote add origin https://github.com/YOUR_USERNAME/persona-agent-system.git
git push -u origin master
```

## Option 2: Manual Setup (GitHub Web UI)

1. Go to https://github.com/new
2. Repository name: `persona-agent-system`
3. Description: "Multi-persona knowledge bank with PydanticAI and RAG - Create AI agents from any expert's public content"
4. Visibility: Public (or Private if preferred)
5. Do NOT initialize with README (we already have one)
6. Click "Create repository"

7. Add remote and push:
```bash
cd /home/samuel/workspace/persona-agent-system
git remote add origin https://github.com/YOUR_USERNAME/persona-agent-system.git
git branch -M main  # Rename master to main (GitHub default)
git push -u origin main
```

## Update Plan Document

After creating the repo, update `.agents/plans/persona-agent-system-mvp.md`:

Replace:
```
**GitHub Repository**: https://github.com/yourusername/persona-agent-system (update with actual repo URL)
```

With your actual repo URL:
```
**GitHub Repository**: https://github.com/YOUR_ACTUAL_USERNAME/persona-agent-system
```

## Update Archon Project

Once repo is created:

```bash
# Update Archon project with actual GitHub URL
# Use Archon MCP tools or update via your remote-coding-agent
```
