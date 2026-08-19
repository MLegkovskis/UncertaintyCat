import { ApiClient } from "@uncertaintycat/contracts";

export const api = new ApiClient();

export async function readTextStream(
  response: Response,
  onChunk: (text: string) => void,
) {
  if (!response.body) throw new Error("Streaming response has no body.");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    onChunk(decoder.decode(value, { stream: true }));
  }
}
