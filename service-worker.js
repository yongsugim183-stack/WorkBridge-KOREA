const CACHE_NAME = "workbridge-v3";

// HTML 페이지는 캐싱하지 않음 - 항상 최신 버전 로드
const HTML_PATHS = ["/", "/board", "/emergency", "/contacts"];

// 설치: 이전 캐시 즉시 교체
self.addEventListener("install", event => {
  self.skipWaiting();
});

// 활성화: 이전 캐시 전체 삭제
self.addEventListener("activate", event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});

// HTML은 항상 네트워크, 정적 파일(아이콘 등)만 캐시
self.addEventListener("fetch", event => {
  const url = new URL(event.request.url);

  // API 요청 → 항상 네트워크
  if (url.pathname.startsWith("/api/")) {
    event.respondWith(fetch(event.request));
    return;
  }

  // HTML 페이지 → 항상 네트워크 (캐시 안 함)
  if (HTML_PATHS.includes(url.pathname) || event.request.headers.get("accept")?.includes("text/html")) {
    event.respondWith(fetch(event.request));
    return;
  }

  // 정적 자산(아이콘, manifest 등) → 캐시 우선, 없으면 네트워크
  event.respondWith(
    caches.match(event.request).then(cached => {
      if (cached) return cached;
      return fetch(event.request).then(res => {
        const clone = res.clone();
        caches.open(CACHE_NAME).then(cache => cache.put(event.request, clone));
        return res;
      });
    })
  );
});
