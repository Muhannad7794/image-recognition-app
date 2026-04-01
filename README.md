# Step-by-Step Guide: Building and Deploying a Custom Block with Edge Impulse CLI

## Why to Build a Custom Block?
### The Default Block Limitation
**Default Behavior:** Edge Impulse’s standard "Image" DSP block flattens 2D/3D data into a 1D array.

**The Conflict:** High-performance models (like those using GlobalAveragePooling2D) require a 3D input shape (H,W,C).

**The Error:** Flattened features detected, but model expects image-like input.

**The Solution** A Custom Dockerized DSP Block that performs OpenCV-based resizing and normalization while preserving the 3D spatial structure via a specific output_config handshake.

---

## 1️⃣ Environment Setup

Ensure you have the following installed on your local machine:

- Node.js & NPM
- Docker Desktop
- Edge Impulse CLI:

```bash
npm install -g edge-impulse-cli

```

## 2️⃣ Connect to Edge Impulse

Log in to your Edge Impulse account and connect your project:

### Initialize the Block (Login & Project Link)
```bash
edge-impulse-blocks init
```
**Wizard Selections:**

***Login:*** Enter your email and password if prompted.

***Owner:*** Select your username or organization.

***Type:*** Choose DSP block.

***Action:*** Choose Create a new block.

***Name:*** Enter OpenCV Preprocessor.

***Description:*** Enter 3D Image Preprocessor for CNNs.

This creates a hidden .ei-block-config file. This file contains the unique id and organizationId needed for the CI/CD pipeline. Commit this file to your repo.

### Clear old sessions
```bash
edge-impulse-blocks --clean
```
### log in to Edge Impulse
```bash
edge-impulse-login
```
### List your projects to get the project ID
```bash
edge-impulse-projects
```
### Connect to your project using the project ID
```bash
edge-impulse-connect <project-id>
```

## 3️⃣ Build and Push the the Custom Block

### Building and Pushing (Manual Test)
Before trusting the automation, run a manual push to verify the Docker build:

```bash
edge-impulse-blocks --push
```

### Setting up GitHub Actions CI/CD
**A. Add Repository Secrets**

- Go to your GitHub Repo > Settings > Secrets and variables > Actions.

- Add a new secret:

  - Name: EI_API_KEY

  - Value: (Find this in Edge Impulse Studio > Dashboard > Keys)

**B. The CI/CD Workflow**

Ensure your .github/workflows/deploy.yml looks like this:

```yaml
name: Deploy DSP Block
on:
  push:
    branches: [ main ]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
      - run: npm install -g edge-impulse-cli
      - run: edge-impulse-blocks push --api-key ${{ secrets.EI_API_KEY }}
```
Once the code is ready, push the changes to GitHub:

```bash
git push
```

### What Happens During the Push?

1. **Docker Build**: The CLI builds the Docker image using the provided Dockerfile.
2. **Library Verification**: Installs all system depedencies
3. **Health Check**: he CLI spins up the container and pings http://localhost:4446/ to ensure the server is listening.
4. **Registry Upload**: If the health check passes, the image is tagged and pushed to the Edge Impulse registry.

## 4️⃣ Deploying the Custom Block

Once the image is pushed, you can deploy it to your Edge Impulse project:

1. Go to your Edge Impulse project dashboard.
2. Navigate to the "Blocks" section.
3. Click "Add Block" and select your custom block from the list.
4. Configure the block as needed and save your changes.
