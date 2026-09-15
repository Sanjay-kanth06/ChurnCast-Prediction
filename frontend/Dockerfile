# ChurnCast frontend (Next.js).
FROM node:22-alpine

ENV NODE_ENV=production
WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm ci --omit=dev || npm install --omit=dev

COPY . .
RUN npm run build

EXPOSE 3000
CMD ["npm", "run", "start"]
