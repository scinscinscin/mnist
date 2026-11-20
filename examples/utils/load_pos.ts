import { Slex, Token as SlexToken } from "@scinorandex/slex";
import { Sparse, buildProductions } from "@scinorandex/sparse";
import fs from "fs";
import path from "path";
import { PorterStemmer } from "natural";
import { load_model } from "./word2vec/model";
import { Matrix } from "../../src/matrix";
import { Sample, Tensor3D } from "../../src/network";

// prettier-ignore
export enum TokenType {
  EOF,
  IDENTIFIER,
  STRING, INTEGER, FLOAT, NULL, TRUE, FALSE, 
  LEFT_BRACE, RIGHT_BRACE, LEFT_BRACKET, RIGHT_BRACKET, COLON, COMMA,
  SINGLE_LINE_COMMENT, MULTI_LINE_COMMENT,
}

export type TokenMetadata = {};
export type JsonAppendToken = SlexToken<TokenType, TokenMetadata>;

export const toStringifiedTokenType = (type: TokenType) => TokenType[type];
export const lexerGenerator = new Slex<TokenType, TokenMetadata>({
  EOF_TYPE: TokenType.EOF,
  isHigherPrecedence: ({ current, next }) => current === TokenType.IDENTIFIER,
  ignoreTokens: [TokenType.MULTI_LINE_COMMENT, TokenType.SINGLE_LINE_COMMENT],
});

lexerGenerator.addRule("lowercase", "a|b|c|d|e|f|g|h|i|j|k|l|m|n|o|p|q|r|s|t|u|v|w|x|y|z");
lexerGenerator.addRule("uppercase", "A|B|C|D|E|F|G|H|I|J|K|L|M|N|O|P|Q|R|S|T|U|V|W|X|Y|Z");
lexerGenerator.addRule("letter", "${lowercase}|${uppercase}");
lexerGenerator.addRule(
  "symbols",
  "$ | $! | $@ | $# | $$ | $% | $^ | $& | $* | $( | $) | ${ | $[ | $} | $] | $; | $: | $< | $, | $. | $> | $? | $/ | $` | $~ | $- | $_ | $+ | $= | $|",
);
lexerGenerator.addRule("escape_character", "$\\ | $\n | $\t | $\r | $\\$\" | $\\$'");
lexerGenerator.addRule("digit", "0|1|2|3|4|5|6|7|8|9");
lexerGenerator.addRule("character", "${letter}|${digit}|${symbols}|${escape_character}");
lexerGenerator.addRule("identifier", "((${letter}|$_)(${letter}|${digit}|$_)*)^", TokenType.IDENTIFIER);

// handle literal tokens
lexerGenerator.addRule(
  "string_literal",
  "$\"(${character} | $')*$\" | $'(${character} | $\")*$'",
  TokenType.STRING,
  (str) => str.substring(1, str.length - 1).replaceAll("\\n", "\n"),
);

lexerGenerator.addRule("integer_literal", "(${__decimal_digit})+", TokenType.INTEGER);
lexerGenerator.addRule("float_literal", "(${__decimal_digit})+$.(${__decimal_digit})+", TokenType.FLOAT);
lexerGenerator.addRule("null_literal", "null", TokenType.NULL);
lexerGenerator.addRule("true_literal", "true", TokenType.TRUE);
lexerGenerator.addRule("false_literal", "false", TokenType.FALSE);

lexerGenerator.addRule("l_brace", "$[", TokenType.LEFT_BRACKET);
lexerGenerator.addRule("r_brace", "$]", TokenType.RIGHT_BRACKET);
lexerGenerator.addRule("l_curly_brace", "${", TokenType.LEFT_BRACE);
lexerGenerator.addRule("r_curly_brace", "$}", TokenType.RIGHT_BRACE);
lexerGenerator.addRule("colon", "$:", TokenType.COLON);
lexerGenerator.addRule("comma", "$,", TokenType.COMMA);

// handle single and multi line comments
lexerGenerator.addRule("single_line_comment", "$/$/(($\n)!)*", TokenType.SINGLE_LINE_COMMENT);
lexerGenerator.addRule("multi_line_comment", "$/$*(($*)!|($*($/)!))*$*$/", TokenType.MULTI_LINE_COMMENT);

// Node definitions belong here

abstract class BaseNode {}

export class ListNode<T> extends BaseNode {
  constructor(private readonly items: T[]) {
    super();
  }

  add(node: T) {
    this.items.push(node);
    return this;
  }

  get_items() {
    return this.items;
  }

  get_items_reversed() {
    return this.items.toReversed();
  }
}

export type JsonValue = LiteralNode | ArrayNode | ObjectNode;
export class PropertyNode extends BaseNode {
  constructor(
    public readonly key: string,
    public readonly value: JsonValue,
  ) {
    super();
  }
}

type Json = JsonPrimitive | JsonArray | JsonObject;
type JsonPrimitive = string | number | boolean | null;
type JsonArray = Json[];
type JsonObject = { [key: string]: Json };

export class ObjectNode extends BaseNode {
  constructor(public readonly properties: ListNode<PropertyNode>) {
    super();
  }

  add_property(key: string, value: JsonValue) {
    this.properties.add(new PropertyNode(key, value));
    return this;
  }

  to_json(): Json {
    const ret = {} as JsonObject;
    for (const p of this.properties.get_items()) {
      ret[p.key] = p.value.to_json();
    }
    return ret;
  }
}

export class ArrayNode extends BaseNode {
  constructor(public readonly values: ListNode<JsonValue>) {
    super();
  }

  add_value(value: JsonValue) {
    this.values.add(value);
    return this;
  }

  to_json(): Json {
    return this.values.get_items_reversed().map((v) => v.to_json());
  }
}

export class LiteralNode extends BaseNode {
  constructor(public readonly token: JsonAppendToken) {
    super();
  }

  get_type() {
    if (this.token.type === TokenType.STRING) return "string";
    if (this.token.type === TokenType.INTEGER) return "integer";
    if (this.token.type === TokenType.FLOAT) return "float";
    if (this.token.type === TokenType.NULL) return "null";
    if (this.token.type === TokenType.TRUE || this.token.type === TokenType.FALSE) return "boolean";
    throw new Error("Invariant: Literal node should not be created for non-literal token. " + this.token.type);
  }

  get_string_value() {
    if (this.token.type !== TokenType.STRING)
      throw new Error("Invariant: Literal node should be a string. " + this.token.type);
    return this.token.lexeme;
  }

  to_json(): Json {
    if (this.token.type === TokenType.STRING) return this.get_string_value();
    if (this.token.type === TokenType.INTEGER) return parseInt(this.token.lexeme);
    if (this.token.type === TokenType.FLOAT) return parseFloat(this.token.lexeme);
    if (this.token.type === TokenType.NULL) return null;
    if (this.token.type === TokenType.TRUE) return true;
    if (this.token.type === TokenType.FALSE) return false;
    throw new Error("Invariant: Literal node should not be created for non-literal token. " + this.token.type);
  }
}

type Reducer = (bag: any) => BaseNode;
const Reducers = {
  program: (bag: { objects?: ListNode<ObjectNode> }) => bag.objects ?? new ListNode([]),
  object_list: (bag: { object: ObjectNode; rest?: ListNode<ObjectNode> }) =>
    bag.rest == undefined ? new ListNode([bag.object]) : bag.rest.add(bag.object),

  literal_value: (bag: { literal_value: JsonAppendToken }) => new LiteralNode(bag.literal_value),
  array_value: (bag: { element_list?: ListNode<JsonValue> }) => new ArrayNode(bag.element_list ?? new ListNode([])),
  element_list: (bag: { value: JsonValue; rest?: ListNode<JsonValue> }) =>
    bag.rest == undefined ? new ListNode([bag.value]) : bag.rest.add(bag.value),

  object_value: (bag: { object: ObjectNode }) => bag.object,
  object: (bag: { property_list?: ListNode<PropertyNode> }) => new ObjectNode(bag.property_list ?? new ListNode([])),
  property_list: (bag: { property: PropertyNode; rest?: ListNode<PropertyNode> }) =>
    bag.rest == undefined ? new ListNode([bag.property]) : bag.rest.add(bag.property),
  property: (bag: { key: JsonAppendToken; value: JsonValue }) => new PropertyNode(bag.key.lexeme, bag.value),
} as { [key: string]: Reducer };

const stem = (strs: string[]) => strs.map((x) => PorterStemmer.stem(x));

const GRAMMAR_FILE = path.join(process.cwd(), "./examples/utils/json_append_grammar.txt");

// This variable sets whether to only include words
const TRY_STEMMING_WORDS = false;
export function load_pos(p = "./datasets/pos/train.json") {
  const str = fs.readFileSync(path.join(process.cwd(), p), { encoding: "utf-8" });
  const lexer = lexerGenerator.compile().generate(str, () => ({}));
  const productions = buildProductions(fs.readFileSync(GRAMMAR_FILE, "utf8"));

  const parser_generator = Sparse.fromProductions({ productions, toStringifiedTokenType });
  const parser = parser_generator.generate(lexer, {
    reducer: ({ bag, name }) => {
      const reducer = Reducers[name ?? ""];
      if (reducer != null) return reducer(bag);
      throw new Error("Invariant: Reducer should not be null. " + name);
    },
  });

  const vectors_path = path.join(process.cwd(), "./examples/utils/word2vec/vectors.txt");
  const word2vec = load_model(vectors_path);

  const parse_result = parser.parse();
  const root_node = parse_result.result as ListNode<ObjectNode>;

  const pos_labels = new Set<string>();
  const raw_samples = [] as [number[][], string[]][];

  for (const p of root_node.get_items_reversed()) {
    const sample = p.to_json() as { labels: string[]; words: string[] };
    // const stemmed_words = stem(sample.words);

    const word_vectors = [] as number[][];
    const tags = [] as string[];

    for (let i = 0; i < sample.words.length; i++) {
      // Try to get the vector of the unstemmed word
      let vector = word2vec.get_vector(sample.words[i]);

      // If no vector found and TRY_STEMMING is true then try to get the vector from the stemmed version
      // Generally this should lead to lower accuracy since verbs can be stemmed to only nouns while their tag
      // is unchanged.
      if (vector == null && TRY_STEMMING_WORDS) vector = word2vec.get_vector(PorterStemmer.stem(sample.words[i]));
      if (vector == null) continue;

      word_vectors.push(vector.to_number_array());
      tags.push(sample.labels[i]);
      pos_labels.add(sample.labels[i]);
    }

    raw_samples.push([word_vectors, tags]);
  }

  const pos_label_incides = [...pos_labels];
  const ret = [] as Sample[];

  for (const [vectors, labels] of raw_samples) {
    const word_embeddings = [] as Matrix[];
    const label_vectors = [] as Matrix[];

    for (let i = 0; i < labels.length; i++) {
      const vector = Matrix.from_existing([vectors[i]]);

      const index = pos_label_incides.indexOf(labels[i]);
      const one_shot = Matrix.init(1, pos_label_incides.length);
      one_shot.arr[0][index] = 1;

      word_embeddings.push(vector);
      label_vectors.push(one_shot);
    }

    ret.push({ input: new Tensor3D(word_embeddings), label: new Tensor3D(label_vectors) });
  }

  return { samples: ret, tags: pos_label_incides };
}
