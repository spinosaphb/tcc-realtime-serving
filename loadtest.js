import http from 'k6/http';
import { check, sleep } from 'k6';

// Define your six possible payloads
const payloads = [
    { /* Payload 1 */ },
    { /* Payload 2 */ },
    { /* Payload 3 */ },
    { /* Payload 4 */ },
    { /* Payload 5 */ },
    { /* Payload 6 */ },
];

export let options = {
    vus: 10, // number of virtual users
    duration: '30s', // duration of the test
};

export default function () {
    let url = 'http://localhost:8000/predict';

    // Randomly select a payload from the array
    let randomPayload = payloads[Math.floor(Math.random() * payloads.length)];

    let headers = {
        'Content-Type': 'application/json',
    };

    let res = http.post(url, JSON.stringify(randomPayload), { headers: headers });

    check(res, {
        'status is 200': (r) => r.status === 200,
    });

    sleep(1); // adjust the sleep time based on your scenario
}
