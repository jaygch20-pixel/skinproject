import axios from "axios";

const API_BASE_URL =
  window.__API_BASE_URL__ ||
  import.meta.env.VITE_API_BASE_URL ||
  (import.meta.env.DEV ? "/api" : "https://skinproject-lpnv.onrender.com/api");
const api = axios.create({ baseURL: API_BASE_URL });

api.interceptors.response.use(undefined, (error) => {
  const detail = error.response?.data?.detail;
  if (typeof detail === "string") {
    error.message = detail;
  } else if (Array.isArray(detail)) {
    error.message = detail.map((item) => item.msg).filter(Boolean).join("; ");
  }
  return Promise.reject(error);
});

export async function checkHealth() {
  const { data } = await api.get("/health");
  return data;
}

export async function getModelInfo() {
  const { data } = await api.get("/model-info");
  return data;
}

export async function getEvaluationMetrics() {
  const { data } = await api.get("/results/metrics");
  return data;
}

export async function predictWithFile(file) {
  const formData = new FormData();
  formData.append("file", file);
  
  const { data } = await api.post("/predict", formData);
  return data;
}

export async function predictWithBase64(base64Image, options = {}) {
  const { data } = await api.post("/predict", {
    image: base64Image,
    ...options
  });
  return data;
}

export function getChartUrl(chartName) {
  return `${API_BASE_URL}/results/charts/${chartName}`;
}
