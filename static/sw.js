// AI Playground Service Worker
const CACHE_NAME = 'ai-playground-v1.0.0';
const urlsToCache = [
    '/',
    '/static/style.css',
    '/static/favicon.ico',
    '/static/favicon-32x32.png',
    '/static/favicon-16x16.png',
    '/static/android-chrome-192x192.png',
    '/static/android-chrome-512x512.png',
    '/static/site.webmanifest',
    'https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css',
    'https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js',
    'https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css',
    '/linear_regression',
    '/logistic_regression',
    '/lgbm_playground',
    '/csv_processor',
    '/csv_processor_datasets'
];

// Install event - cache resources
self.addEventListener('install', event => {
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then(cache => {
                console.log('Opened cache');
                // キャッシュに失敗したファイルをスキップ
                return Promise.allSettled(
                    urlsToCache.map(url => 
                        cache.add(url).catch(error => {
                            console.warn(`Failed to cache ${url}:`, error);
                            return null;
                        })
                    )
                );
            })
    );
});

// Fetch event - serve from cache, fallback to network
self.addEventListener('fetch', event => {
    event.respondWith(
        caches.match(event.request)
            .then(response => {
                // Return cached version or fetch from network
                return response || fetch(event.request);
            })
    );
});

// Activate event - clean up old caches
self.addEventListener('activate', event => {
    event.waitUntil(
        caches.keys().then(cacheNames => {
            return Promise.all(
                cacheNames.map(cacheName => {
                    if (cacheName !== CACHE_NAME) {
                        console.log('Deleting old cache:', cacheName);
                        return caches.delete(cacheName);
                    }
                })
            );
        })
    );
});

// Background sync for offline functionality
self.addEventListener('sync', event => {
    if (event.tag === 'background-sync') {
        event.waitUntil(doBackgroundSync());
    }
});

// Push notification handling
self.addEventListener('push', event => {
    const options = {
        body: event.data ? event.data.text() : 'AI Playgroundからの通知です',
        icon: '/static/android-chrome-192x192.png',
        badge: '/static/favicon-32x32.png',
        vibrate: [100, 50, 100],
        data: {
            dateOfArrival: Date.now(),
            primaryKey: 1
        },
        actions: [
            {
                action: 'explore',
                title: 'アプリを開く',
                icon: '/static/icons/open-app.png'
            },
            {
                action: 'close',
                title: '閉じる',
                icon: '/static/icons/close.png'
            }
        ]
    };

    event.waitUntil(
        self.registration.showNotification('AI Playground', options)
    );
});

// Notification click handling
self.addEventListener('notificationclick', event => {
    event.notification.close();

    if (event.action === 'explore') {
        event.waitUntil(
            clients.openWindow('/')
        );
    }
});

// Background sync function
function doBackgroundSync() {
    // Implement background sync logic here
    console.log('Background sync completed');
} 