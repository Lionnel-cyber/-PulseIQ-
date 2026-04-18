import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const UPSTREAM_BASE_URL = (
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000/api/v1"
)
  .replace("://localhost", "://127.0.0.1")
  .replace(/\/$/, "");

export async function GET(
  request: NextRequest,
  context: { params: Promise<{ path: string[] }> }
) {
  const { path } = await context.params;
  const upstreamUrl = `${UPSTREAM_BASE_URL}/${path.join("") ? path.join("/") : ""}${request.nextUrl.search}`;

  try {
    const response = await fetch(upstreamUrl, {
      method: "GET",
      headers: {
        accept: request.headers.get("accept") ?? "application/json",
      },
      cache: "no-store",
    });

    const body = await response.text();

    return new NextResponse(body, {
      status: response.status,
      headers: {
        "content-type":
          response.headers.get("content-type") ?? "application/json",
      },
    });
  } catch (error) {
    return NextResponse.json(
      {
        detail:
          error instanceof Error
            ? error.message
            : "Failed to reach the PulseIQ API upstream.",
      },
      { status: 502 }
    );
  }
}
