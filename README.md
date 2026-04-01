# Step-by-Step Guide: Building and Deploying a Custom Block with Edge Impulse CLI

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

```bash
# Clear old sessions
edge-impulse-blocks --clean
# log in to Edge Impulse
edge-impulse-login
# List your projects to get the project ID
edge-impulse-projects
edge-impulse-connect <project-id>
```

## 3️⃣ Build and Push the the Custom Block

From the root of the repository, execute the push command. The CLI will automatically detect the Dockerfile, build the image locally to verify it, and then push it to the Edge Impulse registry.

```bash
edge-impulse-blocks --push
```

OR, use the GitHub action workflow to automate the pocess on every push to the chosen branch.

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
