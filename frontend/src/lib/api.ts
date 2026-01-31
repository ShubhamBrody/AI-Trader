export type ApiError = {
  status: number;
  detail: string;
};

async function maybeRedirectToLogin(res: Response): Promise<boolean> {
  if (res.status !== 401) return false;
  const contentType = res.headers.get('content-type') ?? '';
  if (!contentType.includes('application/json')) return false;
  try {
    const json = await res.clone().json();
    const loginUrl = json?.login_url;
    if (typeof loginUrl === 'string' && loginUrl.length > 0) {
      const next = encodeURIComponent(window.location.href);
      const sep = loginUrl.includes('?') ? '&' : '?';
      window.location.href = `${loginUrl}${sep}next=${next}`;
      return true;
    }
  } catch {
    // ignore
  }
  return false;
}

async function readErrorDetail(res: Response): Promise<string> {
  const contentType = res.headers.get('content-type') ?? '';
  try {
    if (contentType.includes('application/json')) {
      const json = await res.json();
      if (typeof json?.detail === 'string') return json.detail;
      return JSON.stringify(json);
    }
    return await res.text();
  } catch {
    return res.statusText;
  }
}

export async function apiGet<T>(path: string): Promise<T> {
  const res = await fetch(path, { headers: { Accept: 'application/json' } });
  if (!res.ok) {
    if (await maybeRedirectToLogin(res)) {
      throw {
        status: res.status,
        detail: 'Redirecting to login…',
      } satisfies ApiError;
    }
    throw {
      status: res.status,
      detail: await readErrorDetail(res),
    } satisfies ApiError;
  }
  return (await res.json()) as T;
}

export async function apiPost<T>(path: string): Promise<T> {
  const res = await fetch(path, {
    method: 'POST',
    headers: { Accept: 'application/json' },
  });
  if (!res.ok) {
    if (await maybeRedirectToLogin(res)) {
      throw {
        status: res.status,
        detail: 'Redirecting to login…',
      } satisfies ApiError;
    }
    throw {
      status: res.status,
      detail: await readErrorDetail(res),
    } satisfies ApiError;
  }
  return (await res.json()) as T;
}
