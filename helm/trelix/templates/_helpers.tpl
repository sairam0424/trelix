{{/*
Expand the name of the chart.
*/}}
{{- define "trelix.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "trelix.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Chart name and version, for the chart label.
*/}}
{{- define "trelix.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels.
*/}}
{{- define "trelix.labels" -}}
helm.sh/chart: {{ include "trelix.chart" . }}
{{ include "trelix.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels.
*/}}
{{- define "trelix.selectorLabels" -}}
app.kubernetes.io/name: {{ include "trelix.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Service account name.
*/}}
{{- define "trelix.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "trelix.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
PVC name — either the user-supplied existingClaim or this release's own PVC.
*/}}
{{- define "trelix.pvcName" -}}
{{- if .Values.persistence.existingClaim }}
{{- .Values.persistence.existingClaim }}
{{- else }}
{{- include "trelix.fullname" . }}
{{- end }}
{{- end }}

{{/*
Container env that every trelix container shares: the vector-store backend and the
embedder credentials. Kept at the container's own 12-space indentation and included
WITHOUT nindent, so a render with nothing extra set stays byte-identical.
*/}}
{{- define "trelix.storeEnv" }}
            - name: TRELIX_STORE_BACKEND
              value: {{ .Values.store.backend | quote }}
            - name: TRELIX_STORE_HNSW
              value: {{ .Values.store.hnsw | quote }}
            - name: TRELIX_STORE_HNSW_M
              value: {{ .Values.store.hnswM | quote }}
            - name: TRELIX_STORE_HNSW_EF_SEARCH
              value: {{ .Values.store.hnswEfSearch | quote }}
            - name: TRELIX_STORE_BM25_READ_POOL_SIZE
              value: {{ .Values.store.bm25ReadPoolSize | quote }}
            {{- if eq .Values.store.backend "qdrant" }}
            - name: QDRANT_URL
              value: {{ .Values.store.qdrant.url | quote }}
            - name: QDRANT_COLLECTION
              value: {{ .Values.store.qdrant.collection | quote }}
            - name: QDRANT_PREFER_GRPC
              value: {{ .Values.store.qdrant.preferGrpc | quote }}
            - name: QDRANT_TIMEOUT
              value: {{ .Values.store.qdrant.timeout | quote }}
            {{- if .Values.store.qdrant.existingSecretName }}
            - name: QDRANT_API_KEY
              valueFrom:
                secretKeyRef:
                  name: {{ .Values.store.qdrant.existingSecretName }}
                  key: {{ .Values.store.qdrant.existingSecretKey }}
            {{- else if .Values.store.qdrant.apiKey }}
            - name: QDRANT_API_KEY
              valueFrom:
                secretKeyRef:
                  name: {{ include "trelix.fullname" . }}
                  key: QDRANT_API_KEY
            {{- end }}
            {{- end }}
            {{- if eq .Values.store.backend "lance" }}
            - name: LANCE_URI
              value: {{ .Values.store.lance.uri | quote }}
            - name: LANCE_TABLE
              value: {{ .Values.store.lance.table | quote }}
            {{- end }}
            {{- if .Values.embedder.openai.existingSecretName }}
            - name: OPENAI_API_KEY
              valueFrom:
                secretKeyRef:
                  name: {{ .Values.embedder.openai.existingSecretName }}
                  key: {{ .Values.embedder.openai.existingSecretKey }}
            {{- else if .Values.embedder.openai.apiKey }}
            - name: OPENAI_API_KEY
              valueFrom:
                secretKeyRef:
                  name: {{ include "trelix.fullname" . }}
                  key: OPENAI_API_KEY
            {{- end }}
            {{- if .Values.embedder.voyage.existingSecretName }}
            - name: VOYAGE_API_KEY
              valueFrom:
                secretKeyRef:
                  name: {{ .Values.embedder.voyage.existingSecretName }}
                  key: {{ .Values.embedder.voyage.existingSecretKey }}
            {{- else if .Values.embedder.voyage.apiKey }}
            - name: VOYAGE_API_KEY
              valueFrom:
                secretKeyRef:
                  name: {{ include "trelix.fullname" . }}
                  key: VOYAGE_API_KEY
            {{- end }}
            {{- if .Values.embedder.cohere.existingSecretName }}
            - name: COHERE_API_KEY
              valueFrom:
                secretKeyRef:
                  name: {{ .Values.embedder.cohere.existingSecretName }}
                  key: {{ .Values.embedder.cohere.existingSecretKey }}
            {{- else if .Values.embedder.cohere.apiKey }}
            - name: COHERE_API_KEY
              valueFrom:
                secretKeyRef:
                  name: {{ include "trelix.fullname" . }}
                  key: COHERE_API_KEY
            {{- end }}
{{- end }}

{{/*
extraEnv and extraSecretEnv, for every container that takes the user's extra variables.
Same indentation rule as trelix.storeEnv.
*/}}
{{- define "trelix.extraEnv" }}
            {{- range .Values.extraEnv }}
            - name: {{ .name }}
              value: {{ .value | quote }}
            {{- end }}
            {{- range .Values.extraSecretEnv }}
            - name: {{ .name }}
              valueFrom:
                secretKeyRef:
                  name: {{ .secretName }}
                  key: {{ .secretKey }}
            {{- end }}
{{- end }}

{{/*
Host allow-list of the trelix-mcp container (TRELIX_MCP_ALLOWED_HOSTS): the release's Service
names, which is what an in-cluster client puts in `Host`, plus mcp.allowedHosts. The `.svc`
form is listed; the `.svc.cluster.local` one is not (add it through mcp.allowedHosts). The
value is never exactly `*`, so the chart cannot turn the Host check off.
*/}}
{{- define "trelix.mcpAllowedHosts" -}}
{{- $fullname := include "trelix.fullname" . -}}
{{- $names := list $fullname (printf "%s.%s" $fullname .Release.Namespace) (printf "%s.%s.svc" $fullname .Release.Namespace) -}}
{{- join "," (concat $names .Values.mcp.allowedHosts) -}}
{{- end }}

{{/*
Refuse the two mcp.enabled values that cannot work. `helm template` and `helm install` fail
on it; `helm lint` only logs the message and exits 0. Included at the top of deployment.yaml.
*/}}
{{- define "trelix.mcpChecks" -}}
{{- if .Values.mcp.enabled }}
{{- if not (or .Values.mcp.auth.token .Values.mcp.auth.existingSecretName) }}
{{- fail "mcp.enabled needs mcp.auth.token or mcp.auth.existingSecretName: trelix-mcp binds 0.0.0.0 in the pod and exits 2 without a token" }}
{{- end }}
{{- if not .Values.persistence.enabled }}
{{- fail "mcp.enabled needs persistence.enabled: trelix-mcp serves the repositories under persistence.mountPath and has nothing to serve without it" }}
{{- end }}
{{- end }}
{{- end }}
