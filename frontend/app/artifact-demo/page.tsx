import { ArtifactDemoTabs } from "@/components/ArtifactDemoTabs";
import { getWorkflowCases } from "@/lib/api";

export default async function ArtifactDemoPage() {
  const workflowPayload = await getWorkflowCases();

  return <ArtifactDemoTabs workflowCases={workflowPayload?.cases ?? []} />;
}
