//! Generic Query Plannning Library providing APIs
//!
//! * to extract relevant parts from a plan (projection, data sources and filters)
//!
//! * to extract relevant information from a plan
//!
//! * to transform expressions
use std::default::Default;
use std::ops::Range;

use serde::{Deserialize, Serialize};

use substrait::proto;
use substrait::proto::{
    expression,
    expression::{mask_expression::StructItem, reference_segment, RexType, ScalarFunction},
    extensions::{
        simple_extension_declaration::{ExtensionFunction, MappingType},
        SimpleExtensionDeclaration,
    },
    function_argument::ArgType,
    plan_rel::RelType as PlanRelType,
    read_rel::ReadType,
    rel::RelType,
    rel_common::Emit,
    rel_common::EmitKind,
    DdlRel,
    Expression,
    ExtensionMultiRel,
    FetchRel,
    FilterRel,
    FunctionArgument,
    NamedStruct,
    Plan,
    PlanRel,
    ProjectRel,
    ReadRel,
    Rel,
    RelCommon,
    RelRoot,
    SetRel,
    Type,
    UpdateRel,
    // fetch_rel::{OffsetMode, CoundMode}, // substrait versions > 0.48
};

/// UUID to refer to the Squid catalog in queries independent of the name
/// that is used in the client.
pub const SQD_ID: &str = "d40ebe93_89e7_4e92_b6ed_452340d405bb";

/// Names of fields referring to block numbers.
/// TODO: This could be defined in metadata.
pub const BLOCK_NUMBER_FIELD_NAMES: &'static [&'static str] = &["number", "block_number"];

/// Names of fields referring to timestamps.
/// TODO: This could be defined in metadata.
pub const TIMESTAMP_FIELD_NAMES: &'static [&'static str] = &["timestamp", "block_time"];

/// Block number and timestamp ranges to which a query refers
/// (extracted from filters, a.k.a. where clause).
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
pub enum FieldRange {
    BlockNumber(Range<i128>),
    Timestamp(Range<i128>),
}

/// Two Fieldranges are of the same type.
impl FieldRange {
    pub fn buddy(&self, other: &Self) -> bool {
        match self {
            FieldRange::BlockNumber(_) => match other {
                FieldRange::BlockNumber(_) => true,
                _ => false,
            },
            FieldRange::Timestamp(_) => match other {
                FieldRange::Timestamp(_) => true,
                _ => false,
            },
        }
    }
}

/// A Source table referred to any part of a query,
/// like in the projection, the from clause or filters.
#[derive(Clone, Debug)]
pub struct Source {
    /// This is a SQD table (not a local table)
    pub sqd: bool,
    /// Table name as it appears in the query
    pub table_name: String,
    /// Schema name as it appears in the query
    pub schema_name: String,
    /// Index of the first field in this source;
    /// all fields of all tables are referenced in the plan
    /// as if it was one slice. The fields belonging to this tables
    /// are those for which holds: n <= first_field < projection.len().
    pub first_field: usize,
    /// Names of all fields in the table.
    pub fields: Vec<String>,
    /// Projection: a list of indexes into fields.
    pub projection: Vec<usize>,
    /// Filters specific for this table.
    /// They are plugged in by the filter pushdown services.
    pub filter: Option<Expression>,
    /// Block ranges relevant for this table.
    pub blocks: Vec<FieldRange>,
}

impl PartialEq for Source {
    fn eq(&self, other: &Self) -> bool {
        self.sqd == other.sqd
            && self.table_name == other.table_name
            && self.schema_name == other.schema_name
    }
}

impl Eq for Source {}

/// Plural of Source.
pub type Sources = Vec<Source>;

/// Errors that may occur in the generic planning library.
#[derive(Debug, thiserror::Error)]
pub enum PlanErr {
    #[error("cannot process plan: {0}")]
    Plan(String),
    #[error("traversal context error: {0}")]
    Context(String),
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Helper to generate PlanErr.
pub fn plan_err<T>(msg: String) -> Result<T, PlanErr> {
    Err(PlanErr::Plan(msg))
}

/// Types of Relations that are present in the reduced plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RelationType {
    Projection,
    Filter,
    Fetch(i64, i64),
    Other(&'static str),
}

/// A Target Plan is a reduced Plan containing only the skeleton of the query
/// with elements that are relevant for SQD: Projection, Filter, Sources and addtional elements
/// that may be requested by users of the library.
/// This trait is implemented by users of the generic plan library.
pub trait TargetPlan {
    /// End of a branch, like NIL.
    fn empty() -> Self;

    /// Generic relation.
    fn from_relation(
        relt: RelationType,
        exps: &[Expression],
        from: &Rel,
        rel: Self,
    ) -> PlanResult<Self>
    where
        Self: Sized;

    /// A join.
    fn from_join(exps: &[Expression], from: &Rel, left: Self, right: Self) -> PlanResult<Self>
    where
        Self: Sized;

    /// A source (or read relation in substrait parlance).
    fn from_source(source: Source) -> Self;

    /// Get the source for one branch.
    fn get_source(&self) -> Option<&Source>;

    /// Get all sources of the query.
    fn get_sources(&self) -> Vec<Source>;
}

/// Generic plan traversal result.
pub type PlanResult<T> = Result<T, PlanErr>;

/// Transform a plan into a target plan by traversing the substrait planning tree.
/// This is a convenience interface for `traverse_plan`.
pub fn transform_plan<T: TargetPlan>(p: &Plan) -> PlanResult<(TraversalContext, T)> {
    let mut tctx = TraversalContext::new(Default::default());
    let target = traverse_plan(p, &mut tctx)?;
    Ok((tctx, target))
}

/// Plan Traversal generating a Target Plan.
pub fn traverse_plan<T: TargetPlan>(p: &Plan, tctx: &mut TraversalContext) -> PlanResult<T> {
    let Plan {
        version,
        extensions,
        relations,
        ..
    } = p;
    let producer = match version {
        Some(ref v) if &v.producer == "DuckDB" => 1,
        _ => 0,
    };

    tctx.producer = producer;
    tctx.extensions = get_extensions(extensions)?;

    traverse_planrel(tctx, relations)
}

/// Options for filter pushdown:
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FilterPushdownLevel {
    /// Don't pushdown filters
    NoPushdown,
    /// Be cautious, don't use aggressive optimisations!
    Cautious,
    /// With block number extraction
    Extract,
    /// Use experimental features.
    Experimental,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Options {
    pub filter_pushdown: FilterPushdownLevel,
}

impl Default for Options {
    fn default() -> Options {
        Options {
            filter_pushdown: FilterPushdownLevel::Extract,
        }
    }
}

/// The Traversal Context collects information that is valid
/// for the whole planning tree:
pub struct TraversalContext {
    /// Final output names that will eventually be applied as aliases
    pub output_names: Vec<String>,
    /// Output mapping over all sources
    pub outmapping: Option<Vec<i32>>,
    /// Extensions, in particular SQL functions, defined in the plan
    pub extensions: Vec<Ext>,
    /// Producer (with DuckDB, function reference starts with 1)
    pub producer: u8,
    /// This query contains joins (otherwise we can just pushdown all filters)
    pub has_joins: bool,
    /// Filter Pushdown Level and other options.
    pub options: Options,
}

impl TraversalContext {
    /// Create an empty Traversal Context.
    pub fn new(opts: Options) -> TraversalContext {
        TraversalContext {
            output_names: Vec::new(),
            outmapping: None,
            extensions: Vec::new(),
            producer: 0,
            has_joins: false,
            options: opts,
        }
    }

    /// Get the function identifier (e.g. GT, LT, EQ, NE, etc.) from the extension spec.
    pub fn get_fun_from_ext(&self, f: &ScalarFunction) -> PlanResult<Ext> {
        let i = if self.producer == 0 {
            f.function_reference as usize
        } else {
            f.function_reference as usize - 1
        };
        if i >= self.extensions.len() {
            return Err(PlanErr::Context("invalid extension reference".to_string()));
        }
        Ok(self.extensions[i].clone())
    }

    /// Get the function reference by Ext
    pub fn ext_to_reference(&self, ext: &Ext) -> Result<u32, PlanErr> {
        let mut i = 0usize;
        // TODO: add a map
        for x in &self.extensions {
            if x == ext {
                if self.producer != 0 {
                    return Ok((i + 1) as u32);
                } else {
                    return Ok(i as u32);
                }
            }
            i += 1;
        }
        Err(PlanErr::Context("invalid extension".to_string()))
    }
}

fn get_extensions(exts: &Vec<SimpleExtensionDeclaration>) -> Result<Vec<Ext>, PlanErr> {
    let mut v = Vec::new();
    for ext in exts {
        v.push(ext_to_fun(&ext)?);
    }
    Ok(v)
}

fn ext_to_fun(ext: &SimpleExtensionDeclaration) -> Result<Ext, PlanErr> {
    match ext.mapping_type {
        Some(MappingType::ExtensionFunction(ref ext)) => Ok(fun_name_to_ext(ext)),
        _ => plan_err("unknown mapping type".to_string()),
    }
}

fn fun_name_to_ext(ext: &ExtensionFunction) -> Ext {
    let mut s = ext.name.split(':');
    match s.next() {
        Some("gt") => Ext::GT,
        Some("gte") => Ext::GE,
        Some("lt") => Ext::LT,
        Some("lte") => Ext::LE,
        Some("equal") => Ext::EQ,
        Some("not_equal") => Ext::NE,
        Some("and") => Ext::And,
        Some("or") => Ext::Or,
        Some("not") => Ext::Not,
        Some("add") => Ext::Add,
        Some("subtract") => Ext::Sub,
        Some("multiply") => Ext::Mul,
        Some("divide") => Ext::Div,
        _ => Ext::Unknown,
    }
}

/// Function Identifiers (a.k.a. extentsions).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Ext {
    GT,
    GE,
    LT,
    LE,
    EQ,
    NE,
    And,
    Or,
    Not,
    Add,
    Sub,
    Mul,
    Div,
    Unknown,
}

/// Comparison Operators.
pub static COMPARE_OPS: &'static [Ext] = &[Ext::GT, Ext::GE, Ext::LT, Ext::LE, Ext::EQ, Ext::NE];

/// Logical Operators.
pub static LOGICAL_OPS: &'static [Ext] = &[Ext::And, Ext::Or, Ext::Not];

// Handle Substrait toplevel relations
fn traverse_planrel<T: TargetPlan>(
    tctx: &mut TraversalContext,
    pls: &Vec<PlanRel>,
) -> PlanResult<T> {
    for rel in pls {
        match rel.rel_type {
            Some(PlanRelType::Root(ref rel)) => return traverse_root(tctx, rel),
            Some(PlanRelType::Rel(_)) => continue,
            None => continue,
        }
    }
    plan_err("no root found".to_string())
}

fn traverse_root<T: TargetPlan>(tctx: &mut TraversalContext, root: &RelRoot) -> PlanResult<T> {
    tctx.output_names = root.names.clone();
    match root.input {
        Some(ref rel) => traverse_relation(tctx, rel),
        None => plan_err("END!".to_string()),
    }
}

// This is the the heart of the matter.
// In principle we could generate all of that with a macro!
fn traverse_relation<T: TargetPlan>(tctx: &mut TraversalContext, rel: &Rel) -> PlanResult<T> {
    match rel.rel_type {
        // properly handled ...
        Some(RelType::Read(ref r)) => traverse_read(tctx, rel, r),
        Some(RelType::Project(ref p)) => traverse_projection(tctx, rel, p),
        Some(RelType::Filter(ref f)) => traverse_filter(tctx, rel, f),
        Some(RelType::Join(ref j)) => {
            traverse_join_kids(tctx, rel, &j.expression, &j.left, &j.right)
        }
        Some(RelType::HashJoin(ref j)) => traverse_join_kids(tctx, rel, &None, &j.left, &j.right),
        Some(RelType::MergeJoin(ref j)) => traverse_join_kids(tctx, rel, &None, &j.left, &j.right),
        Some(RelType::NestedLoopJoin(ref j)) => {
            traverse_join_kids(tctx, rel, &j.expression, &j.left, &j.right)
        }
        Some(RelType::Cross(ref c)) => traverse_join_kids(tctx, rel, &None, &c.left, &c.right),
        Some(RelType::Fetch(ref f)) => traverse_fetch(tctx, rel, f),

        // pass through ...
        Some(RelType::Ddl(ref d)) => traverse_ddl(tctx, d),
        Some(RelType::Set(ref s)) => traverse_set(tctx, s),
        Some(RelType::ExtensionMulti(ref x)) => traverse_ext_multi(tctx, x),
        Some(RelType::Update(ref u)) => traverse_update(tctx, u),

        Some(RelType::Aggregate(ref a)) => traverse_any(tctx, &a.input),
        Some(RelType::Write(ref w)) => traverse_any(tctx, &w.input),
        Some(RelType::Sort(ref s)) => traverse_any(tctx, &s.input),
        Some(RelType::Window(ref w)) => traverse_any(tctx, &w.input),
        Some(RelType::Exchange(ref x)) => traverse_any(tctx, &x.input),
        Some(RelType::Expand(ref x)) => traverse_any(tctx, &x.input),
        Some(RelType::ExtensionSingle(ref x)) => traverse_any(tctx, &x.input),

        // always leaf
        Some(RelType::ExtensionLeaf(_)) => Ok(T::empty()),
        Some(RelType::Reference(_)) => Ok(T::empty()),

        None => plan_err("END".to_string()),
    }
}

fn traverse_any<T: TargetPlan>(tctx: &mut TraversalContext, r: &Option<Box<Rel>>) -> PlanResult<T> {
    match r {
        Some(ref rel) => traverse_relation(tctx, rel),
        None => Ok(T::empty()),
    }
}

fn traverse_read<T: TargetPlan>(
    _tctx: &mut TraversalContext,
    _parent: &Rel,
    r: &ReadRel,
) -> PlanResult<T> {
    read_to_source(r)
}

fn traverse_fetch<T: TargetPlan>(
    tctx: &mut TraversalContext,
    parent: &Rel,
    f: &FetchRel,
) -> PlanResult<T> {
    let input = match f.input {
        Some(ref rel) => traverse_relation(tctx, rel),
        None => Ok(T::empty()),
    }?;

    /* substrait versions > 0.48
    let off = match f.offset_mode {
        Some(OffsetMode::Offset(n)) => n,
        _ => 0,
    };

    let count = match f.count_mode {
        Some(CountMode::Count(n)) => n,
        _ => 0,
    };
    */

    if f.offset == 0 && f.count == 0 {
        T::from_relation(RelationType::Other("ignored fetch"), &[], parent, input)
    } else {
        T::from_relation(RelationType::Fetch(f.offset, f.count), &[], parent, input)
    }
}

fn traverse_ddl<T: TargetPlan>(tctx: &mut TraversalContext, d: &DdlRel) -> PlanResult<T> {
    match d.view_definition {
        Some(ref rel) => traverse_relation(tctx, rel),
        None => Ok(T::empty()),
    }
}

fn traverse_set<T: TargetPlan>(tctx: &mut TraversalContext, s: &SetRel) -> PlanResult<T> {
    if s.inputs.is_empty() {
        Ok(T::empty())
    } else {
        traverse_relation(tctx, &s.inputs[0])
    } // what about the others?
}

fn traverse_ext_multi<T: TargetPlan>(
    tctx: &mut TraversalContext,
    x: &ExtensionMultiRel,
) -> PlanResult<T> {
    if x.inputs.is_empty() {
        Ok(T::empty())
    } else {
        traverse_relation(tctx, &x.inputs[0])
    } // what about the others?
}

fn traverse_projection<T: TargetPlan>(
    tctx: &mut TraversalContext,
    parent: &Rel,
    p: &ProjectRel,
) -> PlanResult<T> {
    let outmap = match p.common {
        Some(RelCommon { ref emit_kind, .. }) => get_out_mapping(emit_kind),
        None => Vec::with_capacity(0),
    };

    if let None = tctx.outmapping {
        tctx.outmapping = Some(outmap);
    }

    let input = match p.input {
        Some(ref rel) => traverse_relation::<T>(tctx, rel),
        None => plan_err("END".to_string()),
    }?;

    T::from_relation(RelationType::Projection, &p.expressions, parent, input)
}

fn get_out_mapping(emit: &Option<EmitKind>) -> Vec<i32> {
    match emit {
        Some(EmitKind::Emit(Emit { ref output_mapping })) => output_mapping.clone(),
        _ => Vec::with_capacity(0),
    }
}

fn traverse_filter<T: TargetPlan>(
    tctx: &mut TraversalContext,
    parent: &Rel,
    f: &FilterRel,
) -> PlanResult<T> {
    let input = match f.input {
        Some(ref rel) => traverse_relation::<T>(tctx, rel),
        None => Ok(T::empty()),
    }?; // filter may sit directly on read and has then no children

    let exp = match &f.condition {
        Some(ref x) => vec![*x.clone()],
        _ => Vec::with_capacity(0),
    };

    T::from_relation(RelationType::Filter, &exp, parent, input)
}

fn traverse_join_kids<T: TargetPlan>(
    tctx: &mut TraversalContext,
    parent: &Rel,
    exp: &Option<Box<Expression>>,
    left: &Option<Box<Rel>>,
    right: &Option<Box<Rel>>,
) -> PlanResult<T> {
    tctx.has_joins = true;

    let left = if let Some(ref l) = left {
        traverse_relation::<T>(tctx, l)?
    } else {
        T::empty()
    };
    let right = if let Some(ref r) = right {
        traverse_relation::<T>(tctx, r)?
    } else {
        T::empty()
    };
    let exps = match exp {
        Some(x) => vec![*x.clone()],
        None => Vec::with_capacity(0),
    };
    T::from_join(&exps, parent, left, right)
}

fn read_to_source<T: TargetPlan>(r: &ReadRel) -> PlanResult<T> {
    let (ours, schema_name, table_name) = match r.read_type {
        Some(ref t) => read_type_to_triple(t),
        None => plan_err("UNKNOWN READ TYPE".to_string()),
    }?;

    tracing::debug!("Source table name: {}.{}", schema_name, table_name);

    let fields: Vec<String> = match r.base_schema {
        Some(NamedStruct { ref names, .. }) => {
            Ok(names.iter().map(|name| name.to_lowercase()).collect())
        }
        None => plan_err("no fields".to_string()),
    }?;

    let projection = match r.projection {
        Some(ref m) if m.select.is_some() => {
            get_projection(&m.select.as_ref().unwrap().struct_items)?
        }
        _ => Vec::with_capacity(0),
    };

    Ok(T::from_source(Source {
        sqd: ours,
        table_name: table_name,
        schema_name: schema_name,
        first_field: 0,
        fields: fields,
        projection: projection,
        filter: r.filter.as_ref().map(|x| *x.clone()),
        blocks: Vec::with_capacity(0),
    }))
}

fn read_type_to_triple(t: &ReadType) -> PlanResult<(bool, String, String)> {
    match t {
        ReadType::NamedTable(ref n) => {
            if n.names.len() > 1 {
                if n.names[0] == SQD_ID {
                    if n.names.len() > 2 {
                        Ok((true, n.names[1].to_string(), n.names[2].to_string()))
                    } else {
                        Ok((true, "".to_string(), n.names[1].to_string()))
                    }
                } else {
                    Ok((false, "".to_string(), "".to_string()))
                }
            } else {
                Ok((false, "".to_string(), "".to_string()))
            }
        }
        other => plan_err(format!("unknown read type: {:?}", other)),
    }
}

fn get_projection(itms: &Vec<StructItem>) -> PlanResult<Vec<usize>> {
    let mut v = Vec::new();
    for itm in itms {
        if itm.field >= 0 {
            v.push(itm.field as usize);
        } else {
            return plan_err("negative index".to_string());
        }
    }
    Ok(v)
}

// Updates have subqueries which we do not handle properly yet.
// For the time being, we just stop!
fn traverse_update<T: TargetPlan>(_tctx: &mut TraversalContext, _u: &UpdateRel) -> PlanResult<T> {
    Ok(T::empty())
}

/// Expression Transformer: transforms a substrait expression into a `Result` of type `T` or error `E`.
pub trait ExprTransformer<T, E> {
    /// A function that takes a message and transforms it into an error.
    fn err_producer<K>(msg: String) -> Result<K, E>;
    /// Transform a literal. TODO: default implementation!
    fn transform_literal(&self, l: &expression::Literal) -> Result<T, E>;
    /// Transform a selection (i.e. field reference). TODO: default implementation!
    fn transform_selection(
        &self,
        tctx: &TraversalContext,
        source: &Source,
        f: &expression::FieldReference,
    ) -> Result<T, E>;
    /// Transform a function. TODO: default implementation!
    fn transform_fun(
        &self,
        tctx: &TraversalContext,
        source: &Source,
        f: &expression::ScalarFunction,
    ) -> Result<T, E>;
    /// Transform a list (e.g.: where field in (a, b, c)). TODO: default implementation!
    fn transform_list(
        &self,
        tctx: &TraversalContext,
        source: &Source,
        l: &expression::SingularOrList,
    ) -> Result<T, E>;

    /// Transform a cast.
    fn transform_cast(
        &self,
        tctx: &TraversalContext,
        source: &Source,
        c: &expression::Cast,
    ) -> Result<T, E> {
        match &c.input {
            Some(expr) => self.transform_expr(&*expr, source, tctx),
            x => Self::err_producer(format!("cast expression: {:?}", x)),
        }
    }

    /// Transform a generic expression.
    fn transform_expr(
        &self,
        x: &Expression,
        source: &Source,
        tctx: &TraversalContext,
    ) -> Result<T, E> {
        match x.rex_type {
            Some(RexType::Literal(ref l)) => self.transform_literal(l),
            Some(RexType::Selection(ref f)) => self.transform_selection(tctx, source, f),
            Some(RexType::ScalarFunction(ref f)) => self.transform_fun(tctx, source, f),
            Some(RexType::SingularOrList(ref s)) => self.transform_list(tctx, source, s),
            Some(RexType::Cast(ref c)) => self.transform_cast(tctx, source, c),
            _ => Self::err_producer(format!(
                "unsupported substrait expression: {:?}",
                x.rex_type
            )),
        }
    }

    /// Get the name of the field referenced by `r` in `source`.
    fn get_field_name_from_source(source: &Source, r: i32) -> Result<String, E> {
        Ok(source.fields[Self::get_field_index(source, r)?].clone())
    }

    /// Get the index into `source.fields` according to `r` going through `source.projection`.
    fn get_field_index(source: &Source, r: i32) -> Result<usize, E> {
        if r as usize >= source.first_field {
            let i = (r as usize) - source.first_field;
            if i < source.projection.len() {
                let j = source.projection[i];
                tracing::trace!(
                    "field name for {} -> {}: {:?} -- {:?}",
                    i,
                    j,
                    source.fields[j],
                    source.projection
                );
                return Ok(j);
            }
        }
        Self::err_producer(format!(
            "invalid field index {}, expected >= {}",
            r, source.first_field
        ))
    }

    /// Get the arguments of a function.
    fn get_fun_args(
        &self,
        tctx: &TraversalContext,
        source: &Source,
        args: &Vec<FunctionArgument>,
    ) -> Result<Vec<T>, E> {
        let mut v = Vec::new();
        for arg in args {
            match arg.arg_type {
                Some(ArgType::Value(ref exp)) => {
                    let x = self.transform_expr(exp, source, tctx)?;
                    v.push(x);
                }
                Some(_) => continue,
                None => continue,
            }
        }
        Ok(v)
    }
}

// ====================================================================================================================
// Mappers
// Mappers are helpers that abstract away the details of the substrait tree
// but too specific to be included in the Expression Transformer tree.
// The closure `transformer`, typically, contains the trait implementation (`Self`)
// in its context. TODO: implement mappers for all relevant parts of the substrait tree.
// ====================================================================================================================
/// Maps a function on a reference segement.
pub fn map_on_dirref<T, F, P, E>(
    source: &Source,
    r: &expression::ReferenceSegment,
    transformer: F,
    err_producer: P,
) -> Result<T, E>
where
    F: Fn(&Source, i32) -> Result<T, E>,
    P: Fn(String) -> Result<T, E>,
{
    match r.reference_type {
        Some(reference_segment::ReferenceType::StructField(ref f)) => transformer(source, f.field),
        _ => err_producer("unsupported segment type".to_string()),
    }
}

// ====================================================================================================================
// Generators
// Generators create part of a substrait tree.
// ====================================================================================================================
/// Generates a new reference segment with StructField set to `r`
pub fn ref_seg_from_ref(r: usize) -> expression::ReferenceSegment {
    expression::ReferenceSegment {
        reference_type: Some(reference_segment::ReferenceType::StructField(Box::new(
            reference_segment::StructField {
                field: r as i32,
                child: None,
            },
        ))),
    }
}

/// Wraps an expression into function argument
pub fn wrap_in_arg(x: Expression) -> FunctionArgument {
    FunctionArgument {
        arg_type: Some(ArgType::Value(x)),
    }
}

/// Wraps a function argument into an expression.
pub fn get_expr_if_scalar_fun(fa: &FunctionArgument) -> Option<expression::ScalarFunction> {
    match fa.arg_type {
        Some(ArgType::Value(ref exp)) => match exp.rex_type {
            Some(RexType::ScalarFunction(ref f)) => Some(f.clone()),
            _ => None,
        },
        _ => None,
    }
}

/// Creates a function from the blueprint other replacing its arguments by `args`.
pub fn make_scalar_fun_from_other(
    other: &expression::ScalarFunction,
    args: &[FunctionArgument],
) -> expression::ScalarFunction {
    expression::ScalarFunction {
        function_reference: other.function_reference,
        arguments: args.to_vec(),
        options: other.options.clone(),
        output_type: other.output_type.clone(),
        #[allow(deprecated)]
        args: Vec::with_capacity(0),
    }
}

pub fn make_scalar_fun_from_args(
    tctx: &TraversalContext,
    fun: Ext,
    args: &[FunctionArgument],
) -> Result<expression::ScalarFunction, PlanErr> {
    Ok(expression::ScalarFunction {
        function_reference: tctx.ext_to_reference(&fun)?,
        arguments: args.to_vec(),
        options: Vec::with_capacity(0),
        output_type: ext_to_output_type(fun),
        #[allow(deprecated)]
        args: Vec::with_capacity(0),
    })
}

pub fn make_scalar_fun(
    tctx: &TraversalContext,
    fun: Ext,
    args: &[Expression],
) -> Result<expression::ScalarFunction, PlanErr> {
    Ok(expression::ScalarFunction {
        function_reference: tctx.ext_to_reference(&fun)?,
        arguments: args.iter().map(|a| wrap_in_arg(a.clone())).collect(),
        options: Vec::with_capacity(0),
        output_type: ext_to_output_type(fun),
        #[allow(deprecated)]
        args: Vec::with_capacity(0),
    })
}

fn ext_to_output_type(fun: Ext) -> Option<Type> {
    if COMPARE_OPS.contains(&fun) || LOGICAL_OPS.contains(&fun) {
        Some(Type {
            kind: Some(proto::r#type::Kind::Bool(proto::r#type::Boolean {
                type_variation_reference: 0,
                nullability: proto::r#type::Nullability::Nullable.into(),
            })),
        })
    } else {
        None
    }
}

#[cfg(test)]
mod test {
    use super::*;

    static JSON_PLAN_BLOCK_COLS: &str =
        include_str!("../resources/block_plain_with_cols.json");
    static JSON_PLAN_BLOCK_SORT: &str =
        include_str!("../resources/block_simple_with_sort.json");
    static JSON_PLAN_BLOCK_COOL_JOIN: &str =
        include_str!("../resources/block_simple_join.json");
    static JSON_PLAN_BLOCK_SQD_JOIN: &str =
        include_str!("../resources/block_remote_join2.json");

    fn make_block_example_with_cols() -> Plan {
        serde_json::from_str(JSON_PLAN_BLOCK_COLS).unwrap()
    }

    fn make_block_example_with_sort() -> Plan {
        serde_json::from_str(JSON_PLAN_BLOCK_SORT).unwrap()
    }

    fn make_block_join_local_example() -> Plan {
        serde_json::from_str(JSON_PLAN_BLOCK_COOL_JOIN).unwrap()
    }

    fn make_block_join_remote_example() -> Plan {
        serde_json::from_str(JSON_PLAN_BLOCK_SQD_JOIN).unwrap()
    }

    #[derive(Debug, PartialEq, Eq)]
    enum TestTarget {
        Empty,
        Join(Box<TestTarget>, Box<TestTarget>),
        Relation(RelationType, Box<TestTarget>),
        Source(Source),
    }

    impl TestTarget {
        fn get_all_sources(&self, v: &mut Vec<Source>) {
            match self {
                TestTarget::Empty => (),
                TestTarget::Relation(_, kid) => kid.get_all_sources(v),
                TestTarget::Join(left, right) => {
                    right.get_all_sources(v);
                    left.get_all_sources(v);
                }
                TestTarget::Source(src) => v.push(src.clone()),
            }
        }
    }

    impl TargetPlan for TestTarget {
        fn empty() -> Self {
            TestTarget::Empty
        }

        fn from_relation(
            relt: RelationType,
            _exps: &[Expression],
            _from: &Rel,
            rel: TestTarget,
        ) -> PlanResult<Self> {
            Ok(TestTarget::Relation(relt, Box::new(rel)))
        }

        fn from_join(
            _exps: &[Expression],
            _from: &Rel,
            left: TestTarget,
            right: TestTarget,
        ) -> PlanResult<Self> {
            match (left, right) {
                (TestTarget::Empty, TestTarget::Empty) => Ok(TestTarget::Empty),
                (TestTarget::Empty, right) => Ok(TestTarget::Relation(
                    RelationType::Other("reduced join"),
                    Box::new(right),
                )),
                (left, TestTarget::Empty) => Ok(TestTarget::Relation(
                    RelationType::Other("reduced join"),
                    Box::new(left),
                )),
                (left, right) => Ok(TestTarget::Join(Box::new(left), Box::new(right))),
            }
        }

        fn from_source(source: Source) -> Self {
            TestTarget::Source(source)
        }

        fn get_source(&self) -> Option<&Source> {
            match self {
                TestTarget::Empty => None,
                TestTarget::Relation(_, kid) => kid.get_source(),
                TestTarget::Join(left, right) => {
                    if let TestTarget::Empty = **left {
                        right.get_source()
                    } else if let TestTarget::Empty = **right {
                        left.get_source()
                    } else {
                        None
                    }
                }
                TestTarget::Source(src) => Some(src),
            }
        }

        fn get_sources(&self) -> Vec<Source> {
            let mut v = Vec::new();
            self.get_all_sources(&mut v);
            v
        }
    }

    #[test]
    fn test_plan_with_simple_block() {
        let p = make_block_example_with_cols();
        let mut tctx = TraversalContext::new(Default::default());
        let target = traverse_plan::<TestTarget>(&p, &mut tctx).expect("plan resulted in an error");
        assert_eq!(
            target,
            TestTarget::Relation(
                RelationType::Projection,
                Box::new(TestTarget::Relation(
                    RelationType::Filter,
                    Box::new(TestTarget::Source(Source {
                        sqd: true,
                        table_name: "block".to_string(),
                        schema_name: "solana_mainnet".to_string(),
                        first_field: 0,
                        fields: vec![
                            "number".to_string(),
                            "hash".to_string(),
                            "parent_number".to_string(),
                            "parent_hash".to_string(),
                            "height".to_string(),
                            "timestamp".to_string(),
                        ],
                        projection: vec![0, 5],
                        filter: None,
                        blocks: vec![],
                    }))
                ))
            )
        );
    }

    #[test]
    fn test_plan_with_sorted_block() {
        let p = make_block_example_with_sort();
        let mut tctx = TraversalContext::new(Default::default());
        let target = traverse_plan::<TestTarget>(&p, &mut tctx).expect("plan resulted in an error");
        println!("Target: {:?}", target);
        assert_eq!(
            target,
            TestTarget::Relation(
                RelationType::Projection,
                Box::new(TestTarget::Relation(
                    RelationType::Filter,
                    Box::new(TestTarget::Source(Source {
                        sqd: true,
                        table_name: "block".to_string(),
                        schema_name: "solana_mainnet".to_string(),
                        first_field: 0,
                        fields: vec![
                            "number".to_string(),
                            "hash".to_string(),
                            "parent_number".to_string(),
                            "parent_hash".to_string(),
                            "height".to_string(),
                            "timestamp".to_string(),
                        ],
                        projection: vec![0, 5],
                        filter: None,
                        blocks: vec![],
                    }))
                ))
            )
        );
    }

    #[test]
    fn test_plan_with_cool_block_join() {
        let p = make_block_join_local_example();
        let mut tctx = TraversalContext::new(Default::default());
        let target = traverse_plan::<TestTarget>(&p, &mut tctx).expect("plan resulted in an error");
        println!("Target: {:?}", target);
        assert_eq!(
            target,
            TestTarget::Relation(
                RelationType::Projection,
                Box::new(TestTarget::Relation(
                    RelationType::Filter,
                    Box::new(TestTarget::Relation(
                        RelationType::Projection,
                        Box::new(TestTarget::Join(
                            Box::new(TestTarget::Source(Source {
                                sqd: true,
                                table_name: "block".to_string(),
                                schema_name: "solana_mainnet".to_string(),
                                first_field: 0,
                                fields: vec![
                                    "number".to_string(),
                                    "hash".to_string(),
                                    "parent_number".to_string(),
                                    "parent_hash".to_string(),
                                    "height".to_string(),
                                    "timestamp".to_string(),
                                ],
                                projection: vec![0, 5],
                                filter: None,
                                blocks: vec![],
                            })),
                            Box::new(TestTarget::Source(Source {
                                sqd: false,
                                table_name: "".to_string(),
                                schema_name: "".to_string(),
                                first_field: 0,
                                fields: vec!["number".to_string(), "note".to_string(),],
                                projection: vec![0, 1],
                                filter: None,
                                blocks: vec![],
                            }))
                        ))
                    ))
                ))
            )
        );
    }

    #[test]
    fn test_plan_remote_join() {
        let p = make_block_join_remote_example();
        let mut tctx = TraversalContext::new(Default::default());
        let target = traverse_plan::<TestTarget>(&p, &mut tctx).expect("plan resulted in an error");
        println!("Target: {:?}", target);
        assert_eq!(
            target,
            TestTarget::Relation(
                RelationType::Projection,
                Box::new(TestTarget::Relation(
                    RelationType::Filter,
                    Box::new(TestTarget::Relation(
                        RelationType::Projection,
                        Box::new(TestTarget::Join(
                            Box::new(TestTarget::Source(Source {
                                sqd: true,
                                table_name: "block".to_string(),
                                schema_name: "solana_mainnet".to_string(),
                                first_field: 0,
                                fields: vec![
                                    "number".to_string(),
                                    "hash".to_string(),
                                    "parent_number".to_string(),
                                    "parent_hash".to_string(),
                                    "height".to_string(),
                                    "timestamp".to_string(),
                                ],
                                projection: vec![0, 4],
                                filter: None,
                                blocks: vec![],
                            })),
                            Box::new(TestTarget::Source(Source {
                                sqd: true,
                                table_name: "transactions".to_string(),
                                schema_name: "solana_mainnet".to_string(),
                                first_field: 0,
                                fields: vec![
                                    "block_number".to_string(),
                                    "transaction_index".to_string(),
                                    "version".to_string(),
                                    "account_keys".to_string(),
                                    "address_table_lookups".to_string(),
                                    "num_readonly_signed_accounts".to_string(),
                                    "num_readonly_unsigned_accounts".to_string(),
                                    "num_required_signatures".to_string(),
                                    "recent_blockhash".to_string(),
                                    "signatures".to_string(),
                                    "err".to_string(),
                                    "compute_units_consumed".to_string(),
                                    "fee".to_string(),
                                    "loaded_addresses".to_string(),
                                    "has_dropped_log_messages".to_string(),
                                    "fee_payer".to_string(),
                                    "account_keys_size".to_string(),
                                    "address_table_lookups_size".to_string(),
                                    "signatures_size".to_string(),
                                    "loaded_addresses_size".to_string(),
                                    "accounts_bloom".to_string(),
                                ],
                                projection: vec![0, 1],
                                filter: None,
                                blocks: vec![],
                            }))
                        ))
                    ))
                ))
            )
        );
    }
}
