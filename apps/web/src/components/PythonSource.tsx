import { python } from "@codemirror/lang-python";
import CodeMirror from "@uiw/react-codemirror";

import { useTheme } from "./Theme";

export function PythonSource({
  source,
  label = "Read-only Python model source",
}: {
  source: string;
  label?: string;
}) {
  const { theme } = useTheme();
  return (
    <div className="python-source-view">
      <CodeMirror
        value={source}
        readOnly
        editable={false}
        theme={theme}
        extensions={[python()]}
        onCreateEditor={(view) =>
          view.contentDOM.setAttribute("aria-label", label)
        }
        basicSetup={{
          lineNumbers: true,
          foldGutter: false,
          highlightActiveLine: false,
          highlightActiveLineGutter: false,
          autocompletion: false,
          bracketMatching: true,
        }}
      />
    </div>
  );
}
