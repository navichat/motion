class JobA {
    constructor(id, duration = 500) {
        this.id = id;
        this.duration = duration;
        this.type = 'computational';
    }

    async execute(worker) {
        // Simulate computational work
        return new Promise((resolve) => {
            setTimeout(() => {
                resolve({
                    jobId: this.id,
                    result: `JobA ${this.id} completed`,
                    executionTime: this.duration
                });
            }, this.duration);
        });
    }
}

class JobB {
    constructor(id, duration = 700) {
        this.id = id;
        this.duration = duration;
        this.type = 'gpu';
    }

    async execute(worker) {
        // Simulate GPU work
        return new Promise((resolve) => {
            setTimeout(() => {
                resolve({
                    jobId: this.id,
                    result: `JobB ${this.id} completed`,
                    executionTime: this.duration
                });
            }, this.duration);
        });
    }
}

class JobC {
    constructor(id, duration = 600) {
        this.id = id;
        this.duration = duration;
        this.type = 'webnn';
    }

    async execute(worker) {
        // Simulate WebNN work
        return new Promise((resolve) => {
            setTimeout(() => {
                resolve({
                    jobId: this.id,
                    result: `JobC ${this.id} completed`,
                    executionTime: this.duration
                });
            }, this.duration);
        });
    }
}

export { JobA, JobB, JobC };