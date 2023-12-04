import http from 'k6/http';
import { check, sleep } from 'k6';
import { readJson } from 'k6/lib/utils';

const allPayloads = readJson('loadtest/resources/onnx_payloads.json');

export let options = {
    vus: 10, // number of virtual users
    duration: '30s', // duration of the test
};

export default function () {
    let url = 'http://localhost:8000/predict';

    allPayloads.forEach((currentPayload) => {
        let headers = {
            'Content-Type': 'application/json',
        };

        let res = http.post(url, JSON.stringify(currentPayload), { headers: headers });

        check(res, {
            'status is 200': (r) => r.status === 200,
        });

        sleep(1);
    });
}
