"use strict";

const CHECK_NAME = "linked issue";
const TARGET_BRANCH = "main";

const CLOSING_ISSUES_QUERY = `
  query ClosingIssues($owner: String!, $repo: String!, $number: Int!) {
    repository(owner: $owner, name: $repo) {
      pullRequest(number: $number) {
        number
        closingIssuesReferences(first: 100) {
          nodes {
            __typename
            number
            state
            url
            repository {
              nameWithOwner
            }
          }
        }
      }
    }
  }
`;

function requireObject(value, label) {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} is missing or malformed`);
  }
  return value;
}

function evaluateClosingIssues(response, expectedRepository, expectedPullNumber) {
  const root = requireObject(response, "GraphQL response");
  const repository = requireObject(root.repository, "GraphQL repository");
  const pullRequest = requireObject(
    repository.pullRequest,
    `pull request #${expectedPullNumber}`,
  );

  if (pullRequest.number !== expectedPullNumber) {
    throw new Error(
      `GraphQL returned pull request #${String(pullRequest.number)} instead of #${expectedPullNumber}`,
    );
  }

  const connection = requireObject(
    pullRequest.closingIssuesReferences,
    "closingIssuesReferences",
  );
  if (!Array.isArray(connection.nodes)) {
    throw new Error("closingIssuesReferences.nodes is missing or malformed");
  }

  const issues = [];
  for (const rawNode of connection.nodes) {
    const node = requireObject(rawNode, "closing issue node");
    const nodeRepository = requireObject(
      node.repository,
      "closing issue repository",
    );
    if (
      node.__typename === "Issue" &&
      nodeRepository.nameWithOwner === expectedRepository &&
      Number.isInteger(node.number) &&
      node.number > 0 &&
      typeof node.url === "string" &&
      node.url.length > 0
    ) {
      issues.push({
        number: node.number,
        state: node.state,
        url: node.url,
      });
    }
  }

  return {
    ok: issues.length > 0,
    issues,
    totalClosingReferences: connection.nodes.length,
  };
}

function pullRequestIdentity(context) {
  const pullRequest = requireObject(
    context && context.payload && context.payload.pull_request,
    "pull_request event payload",
  );
  const base = requireObject(pullRequest.base, "pull request base");
  const baseRepository = requireObject(base.repo, "pull request base repository");
  const head = requireObject(pullRequest.head, "pull request head");
  const headSha = head.sha;
  const expectedRepository = `${context.repo.owner}/${context.repo.repo}`;

  if (base.ref !== TARGET_BRANCH) {
    throw new Error(
      `refusing to evaluate base branch ${String(base.ref)}; expected ${TARGET_BRANCH}`,
    );
  }
  if (baseRepository.full_name !== expectedRepository) {
    throw new Error(
      `base repository ${String(baseRepository.full_name)} does not match ${expectedRepository}`,
    );
  }
  if (!Number.isInteger(pullRequest.number) || pullRequest.number <= 0) {
    throw new Error("pull request number is missing or malformed");
  }
  if (typeof headSha !== "string" || !/^[0-9a-f]{40}$/.test(headSha)) {
    throw new Error("pull request head SHA is missing or malformed");
  }

  return {
    owner: context.repo.owner,
    repo: context.repo.repo,
    repository: expectedRepository,
    pullNumber: pullRequest.number,
    headSha,
  };
}

async function publishStatus(github, identity, targetUrl, state, description) {
  await github.rest.repos.createCommitStatus({
    owner: identity.owner,
    repo: identity.repo,
    sha: identity.headSha,
    state,
    context: CHECK_NAME,
    description,
    target_url: targetUrl,
  });
}

async function run({ github, context, core }) {
  let identity;
  let targetUrl;
  try {
    identity = pullRequestIdentity(context);
    targetUrl = `${context.serverUrl}/${identity.repository}/actions/runs/${context.runId}`;
    await publishStatus(
      github,
      identity,
      targetUrl,
      "pending",
      "Resolving GitHub closing-issue references",
    );

    const response = await github.graphql(CLOSING_ISSUES_QUERY, {
      owner: identity.owner,
      repo: identity.repo,
      number: identity.pullNumber,
    });
    const result = evaluateClosingIssues(
      response,
      identity.repository,
      identity.pullNumber,
    );

    if (!result.ok) {
      await publishStatus(
        github,
        identity,
        targetUrl,
        "failure",
        "No same-repository closing issue is linked",
      );
      core.setFailed(
        `pull request #${identity.pullNumber} has no same-repository closing issue reference`,
      );
      return;
    }

    const linked = result.issues
      .map((issue) => `#${issue.number} (${String(issue.state).toLowerCase()})`)
      .join(", ");
    await publishStatus(
      github,
      identity,
      targetUrl,
      "success",
      `Closes same-repository issue #${result.issues[0].number}`,
    );
    core.info(`accepted closing issue reference(s): ${linked}`);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    if (identity && targetUrl) {
      try {
        await publishStatus(
          github,
          identity,
          targetUrl,
          "failure",
          "Could not verify a linked issue; gate failed closed",
        );
      } catch (publishError) {
        core.error(
          `could not publish the failure to pull request head ${identity.headSha}: ${String(publishError)}`,
        );
      }
    }
    core.setFailed(`linked-issue verification failed closed: ${message}`);
  }
}

module.exports = {
  CHECK_NAME,
  CLOSING_ISSUES_QUERY,
  TARGET_BRANCH,
  evaluateClosingIssues,
  publishStatus,
  pullRequestIdentity,
  run,
};
