/**
 * Binary Grid Database — Arithmetic Evaluator
 * ============================================
 * Shunting-Yard algorithm + RPN evaluation operating on pure integers.
 *
 * The Scale token (S) is a STORAGE ANNOTATION, not an arithmetic operator.
 * NUM S NUM patterns are resolved to ParsedScaledNumber by resolveScaledNumbers()
 * BEFORE arithmetic evaluation.  The database stores pure integers; decimal
 * arithmetic (scale alignment) happens at the application layer via
 * DecimalArithmetic.
 */

import {
  Token, ParsedToken, ParsedNumber, ParsedOperator,
  ParsedScaledNumber,
} from './types';
import { OPERATOR_SYMBOL, NUMERIC_OPERATORS, NUMERIC_ANNOTATIONS } from './tokens';

// ── Precedence table ──────────────────────────────────────────────────────

const PRECEDENCE: Map<Token, number> = new Map([
  [Token.T_EQ, 1],
  [Token.T_PLUS, 2], [Token.T_MINUS, 2],
  [Token.T_MUL, 3], [Token.T_DIV, 3],
  [Token.T_POW, 4],
]);

const RIGHT_ASSOCIATIVE: Set<Token> = new Set([Token.T_POW, Token.T_EQ]);

// ── Scaled number resolution ──────────────────────────────────────────────

/**
 * Post-process parsed tokens: pair NUM S NUM → ParsedScaledNumber.
 *
 * The S token is a storage annotation, not an arithmetic operator.
 * It travels alongside the preceding number in the token stream.
 * This function consumes S and the following number, replacing the
 * three-token sequence (NUM, OP('S'), NUM) with a single ParsedScaledNumber.
 *
 * Example:
 *   [NUM(-1234), OP('S'), NUM(3)] → [SCALED(-1234 / 10^3 = -1.234)]
 */
export function resolveScaledNumbers(tokens: ParsedToken[]): ParsedToken[] {
  const result: ParsedToken[] = [];
  let i = 0;

  while (i < tokens.length) {
    const t = tokens[i];

    if (
      t.type === 'number' &&
      i + 2 < tokens.length &&
      tokens[i + 1].type === 'operator' &&
      (tokens[i + 1] as ParsedOperator).token === Token.T_SCALE &&
      tokens[i + 2].type === 'number'
    ) {
      const numerator = (t as ParsedNumber).value;
      const scale = (tokens[i + 2] as ParsedNumber).value;
      if (scale < 0) throw new Error(`Scale exponent must be non-negative, got ${scale}`);

      const scaled: ParsedScaledNumber = {
        type: 'scaled',
        numerator,
        scale,
        asFloat: numerator / Math.pow(10, scale),
      };
      result.push(scaled);
      i += 3;
    } else {
      result.push(t);
      i++;
    }
  }

  return result;
}

// ── Expression token extraction ───────────────────────────────────────────

type ExprToken = ParsedNumber | ParsedScaledNumber | ParsedOperator;

function extractExpressionTokens(tokens: ParsedToken[]): ExprToken[] {
  const expr: ExprToken[] = [];
  for (const pt of tokens) {
    if (pt.type === 'number' || pt.type === 'scaled') {
      expr.push(pt as ParsedNumber | ParsedScaledNumber);
    } else if (pt.type === 'operator') {
      const op = pt as ParsedOperator;
      if (!NUMERIC_ANNOTATIONS.has(op.token)) {
        expr.push(op);
      }
    }
  }
  return expr;
}

// ── Evaluation ────────────────────────────────────────────────────────────

/**
 * Evaluate an arithmetic expression from parsed tokens.
 *
 * First resolves NUM S NUM → ParsedScaledNumber, then evaluates
 * arithmetic on their integer numerators.  The S annotation is
 * consumed during resolution — it does not appear as an operator
 * in the shunting-yard stage.
 *
 * Returns the integer result.
 */
export function evaluateParsed(tokens: ParsedToken[]): number {
  // Step 1: Resolve scaled numbers
  const resolved = resolveScaledNumbers(tokens);

  // Step 2: Extract numbers and arithmetic operators
  const exprTokens = extractExpressionTokens(resolved);
  if (exprTokens.length === 0) return 0;

  // Single token
  if (exprTokens.length === 1) {
    const item = exprTokens[0];
    if (item.type === 'number') return item.value;
    if (item.type === 'scaled') return item.numerator;
  }

  // Shunting-yard: infix → RPN
  const outputQueue: ExprToken[] = [];
  const operatorStack: Token[] = [];

  for (const item of exprTokens) {
    if (item.type === 'number' || item.type === 'scaled') {
      outputQueue.push(item);
    } else if (item.type === 'operator') {
      const tok = item.token;
      if (tok === Token.T_LPAREN) {
        operatorStack.push(tok);
      } else if (tok === Token.T_RPAREN) {
        while (operatorStack.length > 0 && operatorStack[operatorStack.length - 1] !== Token.T_LPAREN) {
          const top = operatorStack.pop()!;
          outputQueue.push({ type: 'operator', token: top, symbol: OPERATOR_SYMBOL.get(top) ?? '?' });
        }
        if (operatorStack.length > 0 && operatorStack[operatorStack.length - 1] === Token.T_LPAREN) {
          operatorStack.pop();
        } else {
          throw new Error('Mismatched parentheses');
        }
      } else {
        const prec = PRECEDENCE.get(tok) ?? 0;
        while (
          operatorStack.length > 0 &&
          operatorStack[operatorStack.length - 1] !== Token.T_LPAREN
        ) {
          const top = operatorStack[operatorStack.length - 1];
          const topPrec = PRECEDENCE.get(top) ?? 0;
          if (
            topPrec > prec ||
            (topPrec === prec && !RIGHT_ASSOCIATIVE.has(tok))
          ) {
            operatorStack.pop();
            outputQueue.push({ type: 'operator', token: top, symbol: OPERATOR_SYMBOL.get(top) ?? '?' });
          } else {
            break;
          }
        }
        operatorStack.push(tok);
      }
    }
  }

  while (operatorStack.length > 0) {
    const top = operatorStack.pop()!;
    if (top === Token.T_LPAREN) throw new Error('Mismatched parentheses');
    outputQueue.push({ type: 'operator', token: top, symbol: OPERATOR_SYMBOL.get(top) ?? '?' });
  }

  // Evaluate RPN
  const stack: number[] = [];
  for (const item of outputQueue) {
    if (item.type === 'number') {
      stack.push(item.value);
    } else if (item.type === 'scaled') {
      stack.push(item.numerator);
    } else if (item.type === 'operator') {
      if (stack.length < 2) throw new Error(`Operator '${item.symbol}' requires two operands`);
      const b = stack.pop()!;
      const a = stack.pop()!;

      switch (item.token) {
        case Token.T_PLUS:  stack.push(a + b); break;
        case Token.T_MINUS: stack.push(a - b); break;
        case Token.T_MUL:   stack.push(a * b); break;
        case Token.T_DIV:
          if (b === 0) throw new Error('Division by zero');
          stack.push(Math.trunc(a / b));
          break;
        case Token.T_POW:   stack.push(Math.pow(a, b)); break;
        case Token.T_EQ:    stack.push(a === b ? 1 : 0); break;
        default: throw new Error(`Unknown operator: ${item.symbol}`);
      }
    }
  }

  return stack.length > 0 ? stack[stack.length - 1] : 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Decimal Arithmetic — Application-layer scale alignment
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Application-layer decimal arithmetic using the (numerator, scale) model.
 *
 * The database stores pure integers.  This class provides scale-aware
 * operations — exactly how financial systems handle currency (store cents,
 * display dollars) and scientific systems handle measurements.
 */
export class DecimalArithmetic {
  /**
   * Align two scaled numbers to a common scale (the larger of the two).
   */
  static align(a: ParsedScaledNumber, b: ParsedScaledNumber): [ParsedScaledNumber, ParsedScaledNumber] {
    const maxScale = Math.max(a.scale, b.scale);
    return [
      {
        type: 'scaled',
        numerator: a.numerator * Math.pow(10, maxScale - a.scale),
        scale: maxScale,
        asFloat: a.numerator / Math.pow(10, a.scale),
      },
      {
        type: 'scaled',
        numerator: b.numerator * Math.pow(10, maxScale - b.scale),
        scale: maxScale,
        asFloat: b.numerator / Math.pow(10, b.scale),
      },
    ];
  }

  static add(a: ParsedScaledNumber, b: ParsedScaledNumber): ParsedScaledNumber {
    const [aa, bb] = DecimalArithmetic.align(a, b);
    return {
      type: 'scaled',
      numerator: aa.numerator + bb.numerator,
      scale: aa.scale,
      asFloat: (aa.numerator + bb.numerator) / Math.pow(10, aa.scale),
    };
  }

  static subtract(a: ParsedScaledNumber, b: ParsedScaledNumber): ParsedScaledNumber {
    const [aa, bb] = DecimalArithmetic.align(a, b);
    return {
      type: 'scaled',
      numerator: aa.numerator - bb.numerator,
      scale: aa.scale,
      asFloat: (aa.numerator - bb.numerator) / Math.pow(10, aa.scale),
    };
  }

  static multiply(a: ParsedScaledNumber, b: ParsedScaledNumber): ParsedScaledNumber {
    const num = a.numerator * b.numerator;
    const sc = a.scale + b.scale;
    return { type: 'scaled', numerator: num, scale: sc, asFloat: num / Math.pow(10, sc) };
  }

  static fromFloat(value: number, maxScale: number = 9): ParsedScaledNumber {
    let scale = 0;
    let v = value;
    while (v !== Math.trunc(v) && scale < maxScale) {
      v *= 10;
      scale++;
    }
    // Handle floating-point imprecision
    const numerator = Math.round(v);
    return { type: 'scaled', numerator, scale, asFloat: numerator / Math.pow(10, scale) };
  }
}
