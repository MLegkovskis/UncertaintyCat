import {
  AlertTriangle,
  Check,
  Clock3,
  LoaderCircle,
  XCircle,
} from "lucide-react";

export function StatusBadge({ status }: { status: string }) {
  const normalized = status.replaceAll("_", " ");
  const icon =
    status === "succeeded" ? (
      <Check />
    ) : status === "failed" ? (
      <XCircle />
    ) : status === "partially_succeeded" ? (
      <AlertTriangle />
    ) : status === "running" ? (
      <LoaderCircle className="spin" />
    ) : (
      <Clock3 />
    );
  return (
    <span className={`status-badge status-${status}`}>
      {icon}
      {normalized}
    </span>
  );
}

export function EmptyState({ title, body }: { title: string; body: string }) {
  return (
    <div className="empty-state">
      <span className="empty-cat">ᓚᘏᗢ</span>
      <h3>{title}</h3>
      <p>{body}</p>
    </div>
  );
}
