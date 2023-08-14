const SPACE_RULE = '" "?'

const PRIMITIVE_RULES: { [key: string]: string } = {
  boolean: '("true" | "false") space',
  number:
    '("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? space',
  integer: '("-"? ([0-9] | [1-9] [0-9]*)) space',
  string: ` "\\"" (
        [^"\\\\] |
        "\\\\" (["\\\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
      )* "\\"" space`,
  null: '"null" space',
}

const INVALID_RULE_CHARS_RE = /[^\dA-Za-z-]+/g
const GRAMMAR_LITERAL_ESCAPE_RE = /[\n\r"]/g
const GRAMMAR_LITERAL_ESCAPES = { '\r': '\\r', '\n': '\\n', '"': '\\"' } as {
  [key: string]: string
}

const formatLiteral = (literal: string): string => {
  const escaped = JSON.stringify(literal).replace(
    GRAMMAR_LITERAL_ESCAPE_RE,
    (m) => GRAMMAR_LITERAL_ESCAPES[m] || '',
  )
  return `"${escaped}"`
}

interface PropOrder {
  [key: string]: number
}

// JSON schema to grammar converter (Ref: https://github.com/ggerganov/llama.cpp/blob/master/examples/json-schema-to-grammar.py)
export class SchemaGrammarConverter {
  private _propOrder: PropOrder

  private _rules: Map<string, string>

  constructor(propOrder?: PropOrder) {
    this._propOrder = propOrder || {}
    this._rules = new Map<string, string>();
    this._rules.set('space', SPACE_RULE)
  }

  private addRule(name: string, rule: string): string {
    const escName = name.replace(INVALID_RULE_CHARS_RE, '-');
    let key = escName;

    if (this._rules.has(escName)) {
      if (this._rules.get(escName) === rule) {
        return key;
      }

      let i = 0;
      while (this._rules.has(`${escName}${i}`)) {
        i += 1;
      }
      key = `${escName}${i}`;
    }

    this._rules.set(key, rule);
    return key;
  }

  public visit(schema: any, name?: string): string {
    const schemaType = schema.type
    const ruleName = name || 'root'

    if (schema.oneOf || schema.anyOf) {
      const rule = (schema.oneOf || schema.anyOf)
        .map((altSchema: any, i: number) =>
          this.visit(altSchema, `${name}${name ? '-' : ''}${i}`),
        )
        .join(' | ')

      return this.addRule(ruleName, rule)
    } else if ('const' in schema) {
      return this.addRule(ruleName, formatLiteral(schema.const))
    } else if ('enum' in schema) {
      const rule = schema.enum.map((v: string) => formatLiteral(v)).join(' | ')
      return this.addRule(ruleName, rule)
    } else if (schemaType === 'object' && 'properties' in schema) {
      // TODO: `required` keyword (from python implementation)
      const propOrder = this._propOrder
      const propPairs = Object.entries(schema.properties).sort((a, b) => {
        // sort by position in prop_order (if specified) then by key
        const orderA = propOrder[a[0]] ?? Infinity
        const orderB = propOrder[b[0]] ?? Infinity
        return orderA - orderB || a[0].localeCompare(b[0])
      })

      let rule = '"{" space'
      propPairs.forEach(([propName, propSchema], i) => {
        const propRuleName = this.visit(
          propSchema,
          `${name}${name ? '-' : ''}${propName}`,
        )
        if (i > 0) {
          rule += ' "," space'
        }
        rule += ` ${formatLiteral(propName)} space ":" space ${propRuleName}`
      });
      rule += ' "}" space'

      return this.addRule(ruleName, rule)
    } else if (schemaType === 'array' && 'items' in schema) {
      // TODO `prefixItems` keyword (from python implementation)
      const itemRuleName = this.visit(
        schema.items,
        `${name}${name ? '-' : ''}item`,
      )
      const rule = `"[" space (${itemRuleName} ("," space ${itemRuleName})*)? "]" space`
      return this.addRule(ruleName, rule)
    } else {
      if (!PRIMITIVE_RULES[schemaType]) {
        throw new Error(`Unrecognized schema: ${JSON.stringify(schema)}`)
      }
      return this.addRule(
        ruleName === 'root' ? 'root' : schemaType,
        PRIMITIVE_RULES[schemaType] || '',
      )
    }
  }

  public formatGrammar(): string {
    let grammar = '';
    this._rules.forEach((rule, name) => {
      grammar += `${name} ::= ${rule}\n`;
    });
    return grammar;
  }
}

export const convertJsonSchemaToGrammar = (
  { schema, propOrder }: { schema: any; propOrder?: PropOrder },
): string => {
  const converter = new SchemaGrammarConverter(propOrder)
  converter.visit(schema, '')
  return converter.formatGrammar()
}
