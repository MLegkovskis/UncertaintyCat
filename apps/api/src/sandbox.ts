import { Sandbox } from "@cloudflare/sandbox";

/** Disposable, network-isolated OpenTURNS execution boundary. */
export class IsolatedComputeSandbox extends Sandbox {
  enableInternet = false;
}
